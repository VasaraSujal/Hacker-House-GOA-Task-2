"""Stream MSMARCO-XI, chunk, embed, and index into Qdrant + BM25."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from rag.embeddings.local import LocalEmbeddingProvider
from rag.ingestion.indexer import Indexer
from rag.retrieval.bm25 import BM25Index
from rag.retrieval.qdrant_store import QdrantStore


def main() -> int:
    settings = get_settings()
    configure_logging(settings.log_level)
    log = get_logger("ingest")
    log.info(
        "Starting ingestion",
        extra={
            "mode": settings.ingest_mode,
            "config": settings.dataset_config,
            "split": settings.dataset_split,
            "max_documents": settings.max_documents,
            "chunk_strategy": settings.chunk_strategy,
            "embedding_model": settings.embedding_model,
        },
    )
    embeddings = LocalEmbeddingProvider(
        settings.embedding_model,
        device=settings.embedding_device,
        normalize=settings.embedding_normalize,
        batch_size=settings.embedding_batch_size,
    )
    store = QdrantStore(
        url=settings.qdrant_url,
        collection=settings.qdrant_collection,
        api_key=settings.qdrant_api_key,
        vector_size=embeddings.dimension,
        timeout=settings.qdrant_timeout_s,
    )
    if not store.ping():
        log.error("Qdrant is unavailable", extra={"url": settings.qdrant_url})
        print("Qdrant is not reachable. Start it with: docker compose up -d")
        return 1
    bm25 = BM25Index.load(settings.bm25_index_path)
    indexer = Indexer(settings, embeddings, store, bm25)
    stats = indexer.ingest(progress=print)
    print(
        f"Done. queries={stats.query_records} passages={stats.passages} "
        f"chunks={stats.chunks} batches={stats.batches}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
