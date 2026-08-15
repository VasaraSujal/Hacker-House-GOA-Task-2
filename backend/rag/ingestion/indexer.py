from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable

from app.core.config import Settings
from app.core.exceptions import EmbeddingError, VectorStoreError
from rag.chunking.base import Chunk, Chunker
from rag.chunking.factory import get_chunker
from rag.embeddings.base import EmbeddingProvider
from rag.ingestion.checkpoint import IngestCheckpoint, load_checkpoint, save_checkpoint
from rag.ingestion.dataset_loader import iter_records
from rag.retrieval.bm25 import BM25Index
from rag.retrieval.qdrant_store import QdrantStore

logger = logging.getLogger(__name__)


@dataclass
class IngestStats:
    query_records: int = 0
    passages: int = 0
    chunks: int = 0
    skipped_query_ids: int = 0
    batches: int = 0
    retries: int = 0
    failed_batches: int = 0
    elapsed_s: float = 0.0
    peak_rss_mb: float = 0.0
    embedding_s: float = 0.0
    upsert_s: float = 0.0
    bm25_s: float = 0.0
    errors: list[str] = field(default_factory=list)


def _process_rss_mb() -> float:
    try:
        import psutil

        return psutil.Process().memory_info().rss / (1024 * 1024)
    except Exception:  # noqa: BLE001
        return 0.0


class Indexer:
    def __init__(
        self,
        settings: Settings,
        embeddings: EmbeddingProvider,
        store: QdrantStore,
        bm25: BM25Index,
        chunker: Chunker | None = None,
        *,
        max_batch_retries: int | None = None,
        stop_on_batch_failure: bool = True,
    ) -> None:
        self.settings = settings
        self.embeddings = embeddings
        self.store = store
        self.bm25 = bm25
        self.chunker = chunker or get_chunker(
            settings.chunk_strategy,
            chunk_size=settings.chunk_size,
            overlap=settings.chunk_overlap,
            similarity_threshold=settings.semantic_similarity_threshold,
        )
        self.max_batch_retries = (
            settings.ingest_max_batch_retries if max_batch_retries is None else max_batch_retries
        )
        self.stop_on_batch_failure = stop_on_batch_failure

    def _flush_batch(self, pending_chunks: list[Chunk], stats: IngestStats) -> None:
        last_error: Exception | None = None
        attempts = self.max_batch_retries + 1
        for attempt in range(1, attempts + 1):
            try:
                embed_started = time.perf_counter()
                vectors = self.embeddings.embed_documents([c.text for c in pending_chunks])
                stats.embedding_s += time.perf_counter() - embed_started

                upsert_started = time.perf_counter()
                self.store.upsert_chunks(pending_chunks, vectors.tolist())
                stats.upsert_s += time.perf_counter() - upsert_started

                bm25_started = time.perf_counter()
                self.bm25.add_chunks(pending_chunks)
                stats.bm25_s += time.perf_counter() - bm25_started
                return
            except (EmbeddingError, VectorStoreError, OSError, RuntimeError) as exc:
                last_error = exc
                stats.retries += 1
                logger.warning(
                    "Ingest batch failed; retrying",
                    extra={
                        "batch_id": stats.batches + 1,
                        "attempt": attempt,
                        "max_attempts": attempts,
                        "chunk_count": len(pending_chunks),
                        "error": str(exc),
                    },
                )
                if attempt < attempts:
                    time.sleep(min(2 ** (attempt - 1), 8))
        assert last_error is not None
        raise last_error

    def ingest(self, progress: Callable[[str], None] | None = None) -> IngestStats:
        settings = self.settings
        self.store.ensure_collection()
        checkpoint = load_checkpoint(settings.checkpoint_path)
        processed: set[str] = set()
        stats = IngestStats()
        started = time.perf_counter()
        stats.peak_rss_mb = _process_rss_mb()
        if (
            checkpoint
            and checkpoint.dataset_id == settings.dataset_id
            and checkpoint.config == settings.dataset_config
            and checkpoint.split == settings.dataset_split
        ):
            processed = checkpoint.processed_set()
            stats.skipped_query_ids = len(processed)
            stats.chunks = checkpoint.chunks_upserted
            stats.passages = checkpoint.passages_upserted
            logger.info("Resuming ingestion", extra={"already_processed": len(processed)})

        pending_chunks: list[Chunk] = []
        pending_passages = 0
        pending_query_ids: list[str] = []

        def flush() -> None:
            nonlocal pending_chunks, pending_passages, pending_query_ids
            if not pending_chunks:
                return
            batch_id = stats.batches + 1
            query_range = (
                f"{pending_query_ids[0]}..{pending_query_ids[-1]}" if pending_query_ids else "n/a"
            )
            try:
                self._flush_batch(pending_chunks, stats)
            except Exception as exc:  # noqa: BLE001
                stats.failed_batches += 1
                message = (
                    f"batch_id={batch_id} query_range={query_range} "
                    f"chunks={len(pending_chunks)} error={exc}"
                )
                stats.errors.append(message)
                logger.error(
                    "Ingest batch permanently failed",
                    extra={
                        "batch_id": batch_id,
                        "query_range": query_range,
                        "chunk_count": len(pending_chunks),
                        "retry_count": self.max_batch_retries,
                        "error": str(exc),
                        "timestamp": time.time(),
                    },
                )
                if self.stop_on_batch_failure:
                    raise
                pending_chunks = []
                pending_passages = 0
                pending_query_ids = []
                return

            stats.chunks += len(pending_chunks)
            stats.passages += pending_passages
            stats.batches += 1
            processed.update(pending_query_ids)
            pending_chunks = []
            pending_passages = 0
            pending_query_ids = []

            # Persist a bounded ID window for resume; deterministic Qdrant/BM25 IDs
            # still prevent duplicate points if an older ID is reprocessed.
            ordered = list(processed)
            window = settings.ingest_checkpoint_id_window
            persisted_ids = ordered[-window:] if len(ordered) > window else ordered
            ckpt = IngestCheckpoint(
                dataset_id=settings.dataset_id,
                config=settings.dataset_config,
                split=settings.dataset_split,
                processed_query_ids=persisted_ids,
                passages_upserted=stats.passages,
                chunks_upserted=stats.chunks,
                last_query_id=persisted_ids[-1] if persisted_ids else None,
            )
            save_checkpoint(settings.checkpoint_path, ckpt)
            self.bm25.save(settings.bm25_index_path)

            stats.peak_rss_mb = max(stats.peak_rss_mb, _process_rss_mb())
            stats.elapsed_s = time.perf_counter() - started
            chunks_per_s = stats.chunks / stats.elapsed_s if stats.elapsed_s > 0 else 0.0
            msg = (
                f"Ingest batch {stats.batches}: queries={stats.query_records} "
                f"passages={stats.passages} chunks={stats.chunks} "
                f"speed={chunks_per_s:.1f} chunks/s elapsed={stats.elapsed_s:.1f}s "
                f"rss={stats.peak_rss_mb:.0f}MB"
            )
            logger.info(msg)
            if progress:
                progress(msg)

        records = iter_records(
            dataset_id=settings.dataset_id,
            config=settings.dataset_config,
            split=settings.dataset_split,
            max_documents=settings.max_documents if settings.ingest_mode == "subset" else None,
            ingest_mode=settings.ingest_mode,
            index_english=settings.index_english,
            index_translated=settings.index_translated,
            skip_query_ids=processed,
        )

        for record in records:
            stats.query_records += 1
            pending_query_ids.append(record.query_id)
            pending_passages += len(record.passages)
            for passage in record.passages:
                chunks = self.chunker.chunk(
                    passage.text,
                    document_id=passage.document_id,
                    language=passage.language,
                    metadata={
                        "query_id": passage.query_id,
                        "passage_index": passage.passage_index,
                        "is_selected": passage.is_selected,
                        "source_query": passage.source_query,
                        **passage.metadata,
                    },
                )
                pending_chunks.extend(chunks)
            if len(pending_chunks) >= settings.batch_size:
                flush()

        flush()
        stats.elapsed_s = time.perf_counter() - started
        stats.peak_rss_mb = max(stats.peak_rss_mb, _process_rss_mb())
        logger.info(
            "Ingestion complete",
            extra={
                "queries": stats.query_records,
                "passages": stats.passages,
                "chunks": stats.chunks,
                "mode": settings.ingest_mode,
                "elapsed_s": round(stats.elapsed_s, 3),
                "peak_rss_mb": round(stats.peak_rss_mb, 1),
            },
        )
        return stats
