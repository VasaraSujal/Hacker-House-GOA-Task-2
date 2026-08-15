"""Practical multilingual embedding model comparison on the same corpus/qrels."""

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


def _model_memory_mib(provider: LocalEmbeddingProvider) -> float:
    try:
        return sum(
            parameter.numel() * parameter.element_size()
            for parameter in provider._model.parameters()  # type: ignore[attr-defined]
        ) / (1024**2)
    except Exception:
        return 0.0


def _dedupe_doc_ids(indices: np.ndarray, document_ids: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for index in indices:
        document_id = document_ids[int(index)]
        if document_id not in seen:
            seen.add(document_id)
            output.append(document_id)
    return output


def main() -> int:
    settings = get_settings()
    configure_logging(settings.log_level)
    records = list(
        iter_records(
            dataset_id=settings.dataset_id,
            config=settings.dataset_config,
            split=settings.dataset_split,
            max_documents=settings.eval_queries,
            ingest_mode="subset",
            index_english=settings.index_english,
            index_translated=settings.index_translated,
        )
    )
    chunker = get_chunker(
        settings.chunk_strategy,
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
                    metadata={"is_selected": passage.is_selected},
                )
            )
    query_rows = []
    for record in records:
        for language, query in (("en", record.english_query), (record.target_lang.split("_")[0], record.query)):
            relevant = {p.document_id for p in record.passages if p.language == language and p.is_selected}
            if query and relevant:
                query_rows.append({"query": query, "relevant": relevant})

    rows: list[dict] = []
    model_names = [name.strip() for name in settings.embedding_eval_models.split(",") if name.strip()]
    for model_name in model_names:
        load_started = time.perf_counter()
        provider = LocalEmbeddingProvider(
            model_name,
            device=settings.embedding_device,
            normalize=settings.embedding_normalize,
            batch_size=settings.embedding_batch_size,
        )
        load_ms = (time.perf_counter() - load_started) * 1000
        doc_started = time.perf_counter()
        vectors = provider.embed_documents([chunk.text for chunk in chunks])
        document_embedding_ms = (time.perf_counter() - doc_started) * 1000
        metric_rows = []
        query_latencies = []
        search_latencies = []
        document_ids = [chunk.document_id for chunk in chunks]
        for query_row in query_rows:
            query_started = time.perf_counter()
            query_vector = provider.embed_query(query_row["query"])
            query_latencies.append((time.perf_counter() - query_started) * 1000)
            search_started = time.perf_counter()
            scores = vectors @ query_vector
            indices = np.argsort(scores)[::-1][: max(10, settings.dense_top_k)]
            search_latencies.append((time.perf_counter() - search_started) * 1000)
            metric_rows.append(
                score_ranking(
                    query_row["relevant"],
                    _dedupe_doc_ids(indices, document_ids),
                )
            )
        metrics = aggregate_metrics(metric_rows)
        rows.append(
            {
                "model": model_name,
                "dimension": provider.dimension,
                "parameter_memory_mib": _model_memory_mib(provider),
                "load_ms": load_ms,
                "document_embedding_ms": document_embedding_ms,
                "mean_query_embedding_ms": sum(query_latencies) / len(query_latencies),
                "mean_exact_search_ms": sum(search_latencies) / len(search_latencies),
                **asdict(metrics),
            }
        )
        print(
            f"{model_name}: dim={provider.dimension} recall@10={metrics.recall_at_10:.4f} "
            f"query={rows[-1]['mean_query_embedding_ms']:.2f}ms"
        )

    output_dir = settings.benchmark_output_dir / "retrieval"
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "embedding_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_records": len(records),
        "query_count": len(query_rows),
        "chunk_count": len(chunks),
        "chunk_strategy": settings.chunk_strategy,
    }
    (output_dir / "embedding_report.json").write_text(
        json.dumps({"metadata": metadata, "models": rows}, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# Embedding Model Evaluation",
        "",
        f"- Queries: {len(query_rows)}",
        f"- Chunks: {len(chunks)}",
        "",
        "| Model | Dim | Parameter MiB | Query embed (ms) | Recall@5 | Recall@10 | MRR | nDCG@10 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['dimension']} | {row['parameter_memory_mib']:.1f} | "
            f"{row['mean_query_embedding_ms']:.2f} | {row['recall_at_5']:.4f} | "
            f"{row['recall_at_10']:.4f} | {row['mrr']:.4f} | {row['ndcg_at_10']:.4f} |"
        )
    (output_dir / "embedding_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote embedding reports to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
