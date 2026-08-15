"""Extrapolate storage capacity from the measured development index."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import get_settings
from rag.embeddings.local import LocalEmbeddingProvider
from rag.ingestion.checkpoint import load_checkpoint
from rag.retrieval.bm25 import BM25Index


def _mib(value: float) -> float:
    return value / (1024**2)


def main() -> int:
    settings = get_settings()
    bm25 = BM25Index.load(settings.bm25_index_path)
    stats = bm25.chunk_statistics()
    checkpoint = load_checkpoint(settings.checkpoint_path)
    source_records = len(checkpoint.processed_query_ids) if checkpoint else 0
    if not source_records or not stats["chunk_count"]:
        print("A completed development ingestion checkpoint and BM25 index are required.")
        return 1

    embedding = LocalEmbeddingProvider(
        settings.embedding_model,
        device=settings.embedding_device,
        normalize=settings.embedding_normalize,
        batch_size=settings.embedding_batch_size,
    )
    dimension = embedding.dimension
    chunks_per_record = stats["chunk_count"] / source_records
    vector_bytes_per_chunk = dimension * 4
    text_bytes_per_chunk = stats["text_utf8_bytes"] / stats["chunk_count"]
    bm25_file_bytes = settings.bm25_index_path.stat().st_size
    bm25_bytes_per_chunk = bm25_file_bytes / stats["chunk_count"]

    scale_rows = []
    for chunks in [1_000, 10_000, 50_000, 100_000]:
        raw = chunks * (vector_bytes_per_chunk + text_bytes_per_chunk)
        scale_rows.append(
            {
                "chunks": chunks,
                "raw_vector_mib": _mib(chunks * vector_bytes_per_chunk),
                "raw_text_mib": _mib(chunks * text_bytes_per_chunk),
                "qdrant_estimated_low_mib": _mib(raw * 1.3),
                "qdrant_estimated_high_mib": _mib(raw * 2.0),
                "bm25_estimated_mib": _mib(chunks * bm25_bytes_per_chunk),
            }
        )

    corpus_rows = [
        ("development_500_records", 500),
        ("hindi_validation", 97_941),
        ("all_14_validation", 97_941 * 14),
        # Published train counts are about 0.75–0.78M per language.
        ("full_55gb_snapshot_approx", 11_000_000),
    ]
    corpus_estimates = []
    for name, records in corpus_rows:
        chunks = round(records * chunks_per_record)
        raw = chunks * (vector_bytes_per_chunk + text_bytes_per_chunk)
        corpus_estimates.append(
            {
                "corpus": name,
                "estimated_records": records,
                "estimated_chunks": chunks,
                "raw_vector_gib": chunks * vector_bytes_per_chunk / (1024**3),
                "qdrant_estimated_low_gib": raw * 1.3 / (1024**3),
                "qdrant_estimated_high_gib": raw * 2.0 / (1024**3),
                "bm25_estimated_gib": chunks * bm25_bytes_per_chunk / (1024**3),
            }
        )

    report = {
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "measured_source_records": source_records,
            "measured_chunks": stats["chunk_count"],
            "measured_chunks_per_record": chunks_per_record,
            "embedding_model": settings.embedding_model,
            "embedding_dimension": dimension,
            "bm25_file_bytes": bm25_file_bytes,
            "note": (
                "Qdrant estimates use a 1.3x–2.0x range over raw vectors+text for "
                "payload/index/HNSW overhead. Corpus-scale values are extrapolations."
            ),
        },
        "chunk_scales": scale_rows,
        "corpus_estimates": corpus_estimates,
    }
    output_dir = settings.benchmark_output_dir / "scaling"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "scaling_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "# Scaling and Capacity Estimate",
        "",
        f"Measured base: **{source_records} query records / {stats['chunk_count']} chunks** "
        f"({chunks_per_record:.2f} chunks per record).",
        "",
        "All values below are extrapolations from that measured base, not benchmarked full-corpus usage.",
        "",
        "| Corpus | Est. records | Est. chunks | Raw vectors (GiB) | Qdrant range (GiB) | BM25 (GiB) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in corpus_estimates:
        lines.append(
            f"| {row['corpus']} | {row['estimated_records']:,} | {row['estimated_chunks']:,} | "
            f"{row['raw_vector_gib']:.2f} | {row['qdrant_estimated_low_gib']:.2f}–"
            f"{row['qdrant_estimated_high_gib']:.2f} | {row['bm25_estimated_gib']:.2f} |"
        )
    lines.extend(
        [
            "",
            "Qdrant range applies a 1.3x–2.0x overhead factor to raw vectors plus measured text payload.",
            "Actual HNSW, optimizer, replication, filesystem, and payload-index settings can change this.",
        ]
    )
    (output_dir / "scaling_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote scaling reports to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
