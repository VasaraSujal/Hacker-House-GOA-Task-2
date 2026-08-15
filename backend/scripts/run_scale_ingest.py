"""Isolated MSMARCO-XI ingestion scale measurements.

Never writes to the production collection, BM25 index, or checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import Settings, get_settings
from app.core.logging import configure_logging, get_logger
from rag.embeddings.local import LocalEmbeddingProvider
from rag.ingestion.indexer import Indexer
from rag.retrieval.bm25 import BM25Index
from rag.retrieval.qdrant_store import QdrantStore


def _free_ram_gib() -> float:
    try:
        import psutil

        return psutil.virtual_memory().available / (1024**3)
    except Exception:  # noqa: BLE001
        return 0.0


def _disk_free_gib(path: Path) -> float:
    usage = shutil.disk_usage(path)
    return usage.free / (1024**3)


def _qdrant_collection_points(store: QdrantStore) -> int:
    return store.count()


def _delete_collection(store: QdrantStore) -> None:
    try:
        store._client.delete_collection(store.collection)
    except Exception:  # noqa: BLE001
        pass


def _safe_to_run(records: int, free_ram_gib: float, free_disk_gib: float) -> tuple[bool, str]:
    # Bounds from the measured 500-record / 11,478-chunk index:
    # ~23 chunks/record. Embedding model dominates RAM; BM25 file is ~1.4 KB/chunk.
    estimated_chunks = records * 23
    estimated_bm25_gib = estimated_chunks * 3000 / (1024**3)  # tokenized corpus upper bound
    estimated_peak_extra_gib = 2.2 + estimated_bm25_gib
    if free_ram_gib < max(2.5, estimated_peak_extra_gib * 0.55):
        return False, (
            f"insufficient free RAM for ~{records} records "
            f"(free={free_ram_gib:.2f} GiB, estimated working set≈{estimated_peak_extra_gib:.2f} GiB)"
        )
    if free_disk_gib < 5.0:
        return False, f"insufficient free disk (free={free_disk_gib:.2f} GiB)"
    if records >= 50000 and free_ram_gib < 6.0:
        return False, (
            f"refusing {records} records on this host "
            f"(free={free_ram_gib:.2f} GiB; require >=6 GiB free for 50K+)"
        )
    return True, "ok"


def run_scale(records: int, *, cleanup: bool = True) -> dict:
    base = get_settings()
    configure_logging(base.log_level)
    log = get_logger("scale_ingest")
    free_ram = _free_ram_gib()
    free_disk = _disk_free_gib(base.project_root)
    ok, reason = _safe_to_run(records, free_ram, free_disk)
    if not ok:
        return {
            "records_requested": records,
            "status": "skipped",
            "reason": reason,
            "free_ram_gib": free_ram,
            "free_disk_gib": free_disk,
        }

    scale_root = base.project_root / "data" / "scale" / f"records_{records}"
    scale_root.mkdir(parents=True, exist_ok=True)
    collection = f"hh_goa_rag_scale_{records}"
    settings = Settings(
        ingest_mode="subset",
        max_documents=records,
        batch_size=base.batch_size,
        dataset_id=base.dataset_id,
        dataset_config=base.dataset_config,
        dataset_split=base.dataset_split,
        index_english=base.index_english,
        index_translated=base.index_translated,
        chunk_strategy=base.chunk_strategy,
        chunk_size=base.chunk_size,
        chunk_overlap=base.chunk_overlap,
        embedding_model=base.embedding_model,
        embedding_device=base.embedding_device,
        embedding_normalize=base.embedding_normalize,
        embedding_batch_size=base.embedding_batch_size,
        qdrant_url=base.qdrant_url,
        qdrant_api_key=base.qdrant_api_key,
        qdrant_timeout_s=base.qdrant_timeout_s,
        qdrant_collection=collection,
        bm25_index_path=scale_root / "bm25.pkl",
        checkpoint_path=scale_root / "ingest.json",
        ingest_max_batch_retries=base.ingest_max_batch_retries,
        ingest_checkpoint_id_window=base.ingest_checkpoint_id_window,
        ingest_stop_on_batch_failure=True,
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
        return {"records_requested": records, "status": "error", "reason": "qdrant unavailable"}

    _delete_collection(store)
    if settings.bm25_index_path.exists():
        settings.bm25_index_path.unlink()
    if settings.checkpoint_path.exists():
        settings.checkpoint_path.unlink()

    bm25 = BM25Index()
    indexer = Indexer(settings, embeddings, store, bm25)
    wall_started = time.perf_counter()
    log.info("Starting isolated scale ingest", extra={"records": records, "collection": collection})
    stats = indexer.ingest(progress=print)
    wall_s = time.perf_counter() - wall_started
    qdrant_points = _qdrant_collection_points(store)
    bm25_bytes = settings.bm25_index_path.stat().st_size if settings.bm25_index_path.exists() else 0
    result = {
        "records_requested": records,
        "status": "measured",
        "query_records": stats.query_records,
        "passages": stats.passages,
        "chunks": stats.chunks,
        "qdrant_points": qdrant_points,
        "batches": stats.batches,
        "retries": stats.retries,
        "failed_batches": stats.failed_batches,
        "wall_s": round(wall_s, 3),
        "embedding_s": round(stats.embedding_s, 3),
        "upsert_s": round(stats.upsert_s, 3),
        "bm25_s": round(stats.bm25_s, 3),
        "peak_rss_mb": round(stats.peak_rss_mb, 1),
        "bm25_file_bytes": bm25_bytes,
        "chunks_per_s": round(stats.chunks / wall_s, 2) if wall_s > 0 else 0.0,
        "records_per_s": round(stats.query_records / wall_s, 2) if wall_s > 0 else 0.0,
        "free_ram_gib_before": round(free_ram, 2),
        "free_disk_gib_before": round(free_disk, 2),
        "collection": collection,
        "dimension": embeddings.dimension,
    }
    if cleanup:
        _delete_collection(store)
        if settings.bm25_index_path.exists():
            settings.bm25_index_path.unlink()
        if settings.checkpoint_path.exists():
            settings.checkpoint_path.unlink()
        result["cleaned_up"] = True
    else:
        result["cleaned_up"] = False
    return result


def write_reports(rows: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "note": "Isolated scale ingestions. Production hh_goa_rag index was not modified.",
        },
        "scales": rows,
    }
    (output_dir / "scaling_report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    fieldnames = [
        "records_requested",
        "status",
        "query_records",
        "chunks",
        "qdrant_points",
        "wall_s",
        "peak_rss_mb",
        "bm25_file_bytes",
        "chunks_per_s",
        "records_per_s",
        "reason",
    ]
    with (output_dir / "scaling_report.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    lines = [
        "# Scaling Report",
        "",
        "MEASURED isolated ingestions. Production collection unchanged.",
        "",
        "| Records | Status | Chunks | Wall (s) | Peak RSS (MB) | Chunks/s | BM25 (bytes) |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row.get('records_requested', '')} | {row.get('status', '')} | "
            f"{row.get('chunks', '')} | {row.get('wall_s', '')} | {row.get('peak_rss_mb', '')} | "
            f"{row.get('chunks_per_s', '')} | {row.get('bm25_file_bytes', '')} |"
        )
        if row.get("reason"):
            lines.append(f"|  | skip reason: {row['reason']} |  |  |  |  |  |")
    (output_dir / "scaling_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run isolated ingestion scale tests")
    parser.add_argument(
        "--records",
        nargs="+",
        type=int,
        default=[50, 1000, 10000],
        help="Query-record counts to ingest in isolated collections",
    )
    parser.add_argument("--keep", action="store_true", help="Keep temporary collections/files")
    args = parser.parse_args()
    settings = get_settings()
    rows = []
    for records in args.records:
        rows.append(run_scale(records, cleanup=not args.keep))
        print(json.dumps(rows[-1], indent=2))
    write_reports(rows, settings.benchmark_output_dir / "scaling")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
