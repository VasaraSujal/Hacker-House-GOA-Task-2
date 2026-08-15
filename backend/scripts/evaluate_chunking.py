"""Compare fixed, sentence, semantic, and metadata-aware chunking on one corpus."""

from __future__ import annotations

import csv
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import get_settings
from app.core.logging import configure_logging
from evaluation.retrieval_eval import aggregate_metrics, score_ranking
from rag.chunking.factory import get_chunker
from rag.embeddings.local import LocalEmbeddingProvider
from rag.ingestion.dataset_loader import iter_records
from rag.retrieval.bm25 import BM25Index
from rag.retrieval.fusion import reciprocal_rank_fusion
from rag.retrieval.types import RetrievalResult

STRATEGIES = ["fixed", "sentence", "semantic", "metadata"]


def _dedupe_doc_ids(results: list[RetrievalResult]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for result in results:
        if result.document_id not in seen:
            seen.add(result.document_id)
            output.append(result.document_id)
    return output


def _write(output_dir: Path, rows: list[dict], metadata: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "chunking_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "chunking_report.json").write_text(
        json.dumps({"metadata": metadata, "strategies": rows}, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# Chunking Strategy Evaluation",
        "",
        f"- Source records: {metadata['source_record_count']}",
        f"- Evaluation queries: {metadata['query_count']}",
        f"- Embedding model: `{metadata['embedding_model']}`",
        "",
        "| Strategy | Chunks | Avg chars | Est. raw index (MiB) | Recall@5 | Recall@10 | MRR | nDCG@10 | Mean retrieval (ms) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['strategy']} | {row['chunk_count']} | {row['average_chunk_chars']:.1f} | "
            f"{row['estimated_raw_index_mib']:.2f} | {row['recall_at_5']:.4f} | "
            f"{row['recall_at_10']:.4f} | {row['mrr']:.4f} | {row['ndcg_at_10']:.4f} | "
            f"{row['mean_retrieval_ms']:.2f} |"
        )
    (output_dir / "chunking_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    settings = get_settings()
    configure_logging(settings.log_level)
    record_limit = max(1, settings.eval_queries)
    records = list(
        iter_records(
            dataset_id=settings.dataset_id,
            config=settings.dataset_config,
            split=settings.dataset_split,
            max_documents=record_limit,
            ingest_mode="subset",
            index_english=settings.index_english,
            index_translated=settings.index_translated,
        )
    )
    embeddings = LocalEmbeddingProvider(
        settings.embedding_model,
        device=settings.embedding_device,
        normalize=settings.embedding_normalize,
        batch_size=settings.embedding_batch_size,
    )
    query_rows = []
    for record in records:
        for language, query in (("en", record.english_query), (record.target_lang.split("_")[0], record.query)):
            relevant = {p.document_id for p in record.passages if p.language == language and p.is_selected}
            if query and relevant:
                query_rows.append({"query": query, "language": language, "relevant": relevant})
    query_vectors = embeddings.embed_documents([row["query"] for row in query_rows])

    rows: list[dict] = []
    for strategy in STRATEGIES:
        chunker = get_chunker(
            strategy,
            chunk_size=settings.chunk_size,
            overlap=settings.chunk_overlap,
            similarity_threshold=settings.semantic_similarity_threshold,
        )
        chunks = []
        for record in records:
            for passage in record.passages:
                chunks.extend(
                    chunker.chunk(
                        passage.text,
                        document_id=passage.document_id,
                        language=passage.language,
                        metadata={
                            "query_id": passage.query_id,
                            "is_selected": passage.is_selected,
                            "source_query": passage.source_query,
                        },
                    )
                )
        embed_started = time.perf_counter()
        vectors = embeddings.embed_documents([chunk.text for chunk in chunks])
        indexing_ms = (time.perf_counter() - embed_started) * 1000
        bm25 = BM25Index()
        bm25.add_chunks(chunks)
        metric_rows = []
        retrieval_times = []
        for query_row, query_vector in zip(query_rows, query_vectors):
            started = time.perf_counter()
            scores = vectors @ query_vector
            top_indices = np.argsort(scores)[::-1][: settings.dense_top_k]
            dense = [
                RetrievalResult(
                    text=chunks[int(index)].text,
                    score=float(scores[int(index)]),
                    document_id=chunks[int(index)].document_id,
                    chunk_id=chunks[int(index)].chunk_id,
                    metadata=chunks[int(index)].metadata,
                )
                for index in top_indices
            ]
            sparse = bm25.search(query_row["query"], top_k=settings.bm25_top_k)
            hybrid = reciprocal_rank_fusion(
                [dense, sparse],
                k=settings.rrf_k,
                top_k=settings.hybrid_top_k,
            )
            retrieval_times.append((time.perf_counter() - started) * 1000)
            metric_rows.append(
                score_ranking(query_row["relevant"], _dedupe_doc_ids(hybrid))
            )
        metrics = aggregate_metrics(metric_rows)
        text_bytes = sum(len(chunk.text.encode("utf-8")) for chunk in chunks)
        raw_vector_bytes = len(chunks) * embeddings.dimension * 4
        rows.append(
            {
                "strategy": strategy,
                "chunk_count": len(chunks),
                "average_chunk_chars": (
                    sum(len(chunk.text) for chunk in chunks) / len(chunks) if chunks else 0.0
                ),
                "indexing_ms": indexing_ms,
                "estimated_raw_index_mib": (text_bytes + raw_vector_bytes) / (1024**2),
                **asdict(metrics),
                "mean_retrieval_ms": (
                    sum(retrieval_times) / len(retrieval_times) if retrieval_times else 0.0
                ),
            }
        )
        print(f"{strategy}: chunks={len(chunks)} recall@10={metrics.recall_at_10:.4f}")

    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_record_count": len(records),
        "query_count": len(query_rows),
        "embedding_model": settings.embedding_model,
        "embedding_dimension": embeddings.dimension,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
    }
    output_dir = settings.benchmark_output_dir / "retrieval"
    _write(output_dir, rows, metadata)
    print(f"Wrote chunking reports to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
