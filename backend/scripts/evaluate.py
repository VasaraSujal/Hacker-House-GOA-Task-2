"""Reproducible retrieval ablation over the indexed MSMARCO-XI subset."""

from __future__ import annotations

import csv
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from evaluation.retrieval_eval import RetrievalMetrics, aggregate_metrics, score_ranking
from rag.embeddings.local import LocalEmbeddingProvider
from rag.reranking.cross_encoder import CrossEncoderReranker
from rag.retrieval.bm25 import BM25Index
from rag.retrieval.dense import DenseRetriever
from rag.retrieval.hybrid import HybridRetriever
from rag.retrieval.fusion import reciprocal_rank_fusion, weighted_fusion
from rag.retrieval.qdrant_store import QdrantStore


def _ids(results) -> list[str]:
    """Deduplicate chunks to document IDs while preserving rank."""
    seen: set[str] = set()
    ranked: list[str] = []
    for result in results:
        if result.document_id not in seen:
            seen.add(result.document_id)
            ranked.append(result.document_id)
    return ranked


def _write_reports(output_dir: Path, rows: list[dict], metadata: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "retrieval_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "retrieval_report.json").write_text(
        json.dumps({"metadata": metadata, "systems": rows}, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# Retrieval Ablation Report",
        "",
        f"- Queries: {metadata['query_count']}",
        f"- Index chunks: {metadata['index_chunk_count']}",
        f"- Embedding: `{metadata['embedding_model']}`",
        f"- Reranker: `{metadata['reranker_model']}`",
        "",
        "| Pipeline | Recall@5 | Recall@10 | Precision@5 | MRR | nDCG@10 | Mean latency (ms) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['pipeline']} | {row['recall_at_5']:.4f} | {row['recall_at_10']:.4f} | "
            f"{row['precision_at_5']:.4f} | {row['mrr']:.4f} | {row['ndcg_at_10']:.4f} | "
            f"{row['mean_latency_ms']:.2f} |"
        )
    (output_dir / "retrieval_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    settings = get_settings()
    configure_logging(settings.log_level)
    log = get_logger("eval")

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
    )
    if not store.ping():
        print("Qdrant unavailable. Start docker compose and ingest first.")
        return 1
    bm25 = BM25Index.load(settings.bm25_index_path)
    dense = DenseRetriever(embeddings, store)
    hybrid = HybridRetriever(dense, bm25, fusion_method=settings.fusion_method, rrf_k=settings.rrf_k)
    reranker = CrossEncoderReranker(settings.reranker_model, device=settings.reranker_device)

    queries = bm25.evaluation_queries(limit=settings.eval_queries)
    if not queries:
        print("No labeled queries found in the BM25 index. Re-run ingestion.")
        return 1

    systems = ["dense", "bm25", "hybrid_rrf", "hybrid_weighted", "hybrid_rrf+reranker"]
    metric_buckets = {name: [] for name in systems}
    latency_buckets = {name: [] for name in systems}
    rerank_k = max(10, settings.rerank_top_k)
    for i, row in enumerate(queries, start=1):
        text = row["query"]
        relevant = set(row["relevant_document_ids"])
        hybrid_out = hybrid.search(
            text,
            dense_top_k=settings.dense_top_k,
            bm25_top_k=settings.bm25_top_k,
            hybrid_top_k=settings.hybrid_top_k,
        )
        rrf_started = time.perf_counter()
        rrf = reciprocal_rank_fusion(
            [hybrid_out.dense, hybrid_out.bm25],
            k=settings.rrf_k,
            top_k=settings.hybrid_top_k,
        )
        rrf_ms = (time.perf_counter() - rrf_started) * 1000
        weighted_started = time.perf_counter()
        weighted = weighted_fusion(
            hybrid_out.dense,
            hybrid_out.bm25,
            dense_weight=settings.dense_weight,
            sparse_weight=settings.bm25_weight,
            top_k=settings.hybrid_top_k,
        )
        weighted_ms = (time.perf_counter() - weighted_started) * 1000
        rerank_started = time.perf_counter()
        reranked = reranker.rerank(text, rrf, rerank_k)
        rerank_ms = (time.perf_counter() - rerank_started) * 1000

        ranked = {
            "dense": hybrid_out.dense,
            "bm25": hybrid_out.bm25,
            "hybrid_rrf": rrf,
            "hybrid_weighted": weighted,
            "hybrid_rrf+reranker": reranked,
        }
        latency = {
            "dense": hybrid_out.embedding_ms + hybrid_out.dense_retrieval_ms,
            "bm25": hybrid_out.bm25_ms,
            "hybrid_rrf": hybrid_out.retrieval_wall_ms + rrf_ms,
            "hybrid_weighted": hybrid_out.retrieval_wall_ms + weighted_ms,
            "hybrid_rrf+reranker": hybrid_out.retrieval_wall_ms + rrf_ms + rerank_ms,
        }
        for name in systems:
            metric_buckets[name].append(score_ranking(relevant, _ids(ranked[name])))
            latency_buckets[name].append(latency[name])
        print(f"[{i:03d}/{len(queries)}] {row['language']} {text[:60]}")

    report_rows: list[dict] = []
    for name in systems:
        metrics = aggregate_metrics(metric_buckets[name])
        report_rows.append(
            {
                "pipeline": name,
                **asdict(metrics),
                "mean_latency_ms": sum(latency_buckets[name]) / len(latency_buckets[name]),
            }
        )
    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "query_count": len(queries),
        "index_chunk_count": len(bm25),
        "embedding_model": settings.embedding_model,
        "reranker_model": settings.reranker_model,
        "dense_top_k": settings.dense_top_k,
        "bm25_top_k": settings.bm25_top_k,
        "hybrid_top_k": settings.hybrid_top_k,
        "rerank_eval_top_k": rerank_k,
    }
    output_dir = settings.benchmark_output_dir / "retrieval"
    _write_reports(output_dir, report_rows, metadata)
    print(json.dumps(report_rows, indent=2))
    print(f"\nWrote retrieval reports to {output_dir}")
    log.info("Evaluation complete", extra={"n": len(queries)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
