"""Measure Qdrant search latency at 1K, 10K, and the current index size."""

from __future__ import annotations

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
from evaluation.latency_eval import summarize
from rag.embeddings.local import LocalEmbeddingProvider
from rag.retrieval.bm25 import BM25Index
from rag.retrieval.qdrant_store import QdrantStore


def _copy_collection(client, source: str, target: str, size: int, dimension: int) -> None:
    from qdrant_client.http import models as qmodels

    existing = {collection.name for collection in client.get_collections().collections}
    if target in existing:
        client.delete_collection(target)
    client.create_collection(
        collection_name=target,
        vectors_config=qmodels.VectorParams(size=dimension, distance=qmodels.Distance.COSINE),
    )
    offset = None
    copied = 0
    while copied < size:
        points, offset = client.scroll(
            collection_name=source,
            limit=min(500, size - copied),
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )
        if not points:
            break
        client.upsert(
            collection_name=target,
            points=[
                qmodels.PointStruct(id=point.id, vector=point.vector, payload=point.payload)
                for point in points
            ],
            wait=True,
        )
        copied += len(points)
        if offset is None:
            break
    if copied != size:
        raise RuntimeError(f"Copied {copied}/{size} points into {target}")


def main() -> int:
    settings = get_settings()
    embeddings = LocalEmbeddingProvider(
        settings.embedding_model,
        device=settings.embedding_device,
        normalize=settings.embedding_normalize,
        batch_size=settings.embedding_batch_size,
    )
    source_store = QdrantStore(
        url=settings.qdrant_url,
        collection=settings.qdrant_collection,
        api_key=settings.qdrant_api_key,
        vector_size=embeddings.dimension,
    )
    current_count = source_store.count()
    if current_count < 10_000:
        raise RuntimeError("Scaling benchmark requires at least 10,000 indexed vectors")
    bm25 = BM25Index.load(settings.bm25_index_path)
    queries = [row["query"] for row in bm25.evaluation_queries(limit=100)]
    query_vectors = embeddings.embed_documents(queries)
    client = source_store._client  # Shared benchmark-only administrative client.
    rows = []
    temporary = []
    try:
        for size in [1_000, 10_000, current_count]:
            if size == current_count:
                collection = settings.qdrant_collection
            else:
                collection = f"{settings.qdrant_collection}_scale_{size}"
                temporary.append(collection)
                _copy_collection(client, settings.qdrant_collection, collection, size, embeddings.dimension)
            store = QdrantStore(
                url=settings.qdrant_url,
                collection=collection,
                api_key=settings.qdrant_api_key,
                vector_size=embeddings.dimension,
            )
            for vector in query_vectors[:3]:
                store.search(vector, top_k=settings.dense_top_k)
            latencies = []
            for vector in query_vectors:
                started = time.perf_counter()
                store.search(vector, top_k=settings.dense_top_k)
                latencies.append((time.perf_counter() - started) * 1000)
            rows.append({"vector_count": size, **asdict(summarize("qdrant_ms", latencies))})
            print(f"{size} vectors: Qdrant P50={rows[-1]['p50_ms']:.2f}ms")
    finally:
        for collection in temporary:
            try:
                client.delete_collection(collection)
            except Exception:
                pass

    output_dir = settings.benchmark_output_dir / "scaling"
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "query_count": len(query_vectors),
            "top_k": settings.dense_top_k,
            "embedding_model": settings.embedding_model,
        },
        "scales": rows,
    }
    (output_dir / "scaling_latency.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "# Measured Qdrant Scaling Latency",
        "",
        f"- Queries per scale: {len(query_vectors)}",
        f"- Top K: {settings.dense_top_k}",
        "",
        "| Vectors | P50 (ms) | P70 (ms) | P95 (ms) | P100 (ms) | Mean (ms) |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['vector_count']:,} | {row['p50_ms']:.2f} | {row['p70_ms']:.2f} | "
            f"{row['p95_ms']:.2f} | {row['p100_ms']:.2f} | {row['mean_ms']:.2f} |"
        )
    (output_dir / "scaling_latency.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote scaling latency report to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
