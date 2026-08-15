"""Reproducible full-pipeline latency benchmark.

All values come from actual ``time.perf_counter`` measurements. Model loading is
reported separately from warm request percentiles. No response cache is used.
"""

from __future__ import annotations

import csv
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.dependencies import build_pipeline
from evaluation.latency_eval import LatencyReport, summarize

STAGES = [
    "request_parsing_ms",
    "query_processing_ms",
    "embedding_ms",
    "dense_retrieval_ms",
    "bm25_ms",
    "retrieval_wall_ms",
    "fusion_ms",
    "relevance_guard_ms",
    "reranking_ms",
    "context_building_ms",
    "generation_ms",
    "grounding_ms",
    "rag_core_ms",
    "component_sum_ms",
    "unaccounted_ms",
    "total_ms",
]


def _query_set(pipeline, n: int, max_query_chars: int) -> tuple[list[dict[str, Any]], int]:
    rows = pipeline.hybrid.bm25.evaluation_queries()
    if not rows:
        raise RuntimeError("BM25 index has no labeled evaluation queries; ingest first")
    valid_rows = [
        row
        for row in rows
        if 0 < len(" ".join(row["query"].split())) <= max_query_chars
    ]
    excluded_invalid = len(rows) - len(valid_rows)
    if not valid_rows:
        raise RuntimeError("BM25 index has no valid labeled evaluation queries")
    # Deterministic cycling is only used if the requested sample exceeds unique
    # indexed labeled queries. The report discloses unique_query_count.
    return [valid_rows[i % len(valid_rows)] for i in range(n)], excluded_invalid


def _write_reports(
    output_dir: Path,
    *,
    metadata: dict[str, Any],
    reports: dict[str, LatencyReport],
    raw_rows: list[dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_payload = {
        "metadata": metadata,
        "stages": {name: asdict(report) for name, report in reports.items()},
        "samples": raw_rows,
    }
    (output_dir / "latency_report.json").write_text(
        json.dumps(json_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    fields = ["stage", "n", "min_ms", "mean_ms", "median_ms", "p50_ms", "p70_ms", "p90_ms", "p95_ms", "p100_ms", "max_ms"]
    with (output_dir / "latency_report.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name, report in reports.items():
            values = asdict(report)
            values.pop("name", None)
            writer.writerow({"stage": name, **values})

    sorted_bottlenecks = sorted(
        (
            (name, report.p50_ms)
            for name, report in reports.items()
            if name not in {"total_ms", "component_sum_ms", "unaccounted_ms", "rag_core_ms"}
        ),
        key=lambda item: item[1],
        reverse=True,
    )
    primary = sorted_bottlenecks[0] if sorted_bottlenecks else ("unknown", 0.0)
    secondary = sorted_bottlenecks[1] if len(sorted_bottlenecks) > 1 else ("unknown", 0.0)
    lines = [
        "# RAG Latency Report",
        "",
        f"- Timestamp (UTC): {metadata['timestamp_utc']}",
        f"- Requests: {metadata['request_count']} ({metadata['unique_query_count']} unique)",
        f"- Invalid indexed queries excluded: {metadata['excluded_invalid_query_count']}",
        f"- Index chunks: {metadata['index_chunk_count']}",
        f"- Retrieval mode: {'parallel' if metadata['parallel_retrieval'] else 'sequential'}",
        f"- Response cache: disabled",
        f"- Request parsing: not applicable (direct in-process pipeline harness)",
        f"- Model initialization: {metadata['model_initialization_ms']:.2f} ms (excluded from warm percentiles)",
        f"- Refused responses: {metadata['refused_count']}",
        "",
        "| Component | P50 (ms) | P70 (ms) | P90 (ms) | P95 (ms) | P100 (ms) | Mean (ms) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, report in reports.items():
        lines.append(
            f"| {name} | {report.p50_ms:.2f} | {report.p70_ms:.2f} | "
            f"{report.p90_ms:.2f} | {report.p95_ms:.2f} | {report.p100_ms:.2f} | {report.mean_ms:.2f} |"
        )
    lines.extend(
        [
            "",
            f"Primary measured bottleneck: **{primary[0]}** ({primary[1]:.2f} ms P50).",
            f"Secondary measured bottleneck: **{secondary[0]}** ({secondary[1]:.2f} ms P50).",
            "",
            f"RAG core P50: **{reports['rag_core_ms'].p50_ms:.2f} ms**.",
            f"Full text-to-answer P50: **{reports['total_ms'].p50_ms:.2f} ms**.",
            "",
            "The full total includes ElevenLabs generation. No values are estimated or fabricated.",
        ]
    )
    (output_dir / "latency_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    settings = get_settings()
    configure_logging(settings.log_level)
    log = get_logger("benchmark")
    n = max(1, settings.benchmark_queries)

    init_started = time.perf_counter()
    pipeline = build_pipeline(settings)
    model_initialization_ms = (time.perf_counter() - init_started) * 1000
    query_rows, excluded_invalid_queries = _query_set(
        pipeline,
        n,
        settings.max_query_chars,
    )
    unique_queries = len({row["query"] for row in query_rows})

    warmup = min(settings.benchmark_warmup_queries, len(query_rows))
    print(
        f"Benchmarking {n} requests ({unique_queries} unique), "
        f"warmup={warmup}, generation={settings.benchmark_include_generation}"
    )
    if not settings.benchmark_include_generation:
        raise RuntimeError(
            "BENCHMARK_INCLUDE_GENERATION=false is not supported by the full-pipeline benchmark; "
            "use scripts/evaluate.py for retrieval-only latency"
        )
    for row in query_rows[:warmup]:
        pipeline.run(row["query"])

    buckets: dict[str, list[float]] = {stage: [] for stage in STAGES}
    raw_rows: list[dict[str, Any]] = []
    refused = 0
    for i, row in enumerate(query_rows, start=1):
        wall_started = time.perf_counter()
        result = pipeline.run(row["query"])
        wall_ms = (time.perf_counter() - wall_started) * 1000
        latency = result.latency.model_dump()
        if not latency.get("total_ms"):
            latency["total_ms"] = wall_ms
        for stage in STAGES:
            buckets[stage].append(float(latency.get(stage, 0.0)))
        refused += int(result.refused)
        raw_rows.append(
            {
                "sequence": i,
                "query_id": row["query_id"],
                "language": row["language"],
                "query": row["query"],
                "refused": result.refused,
                "grounded": result.grounded,
                "source_count": len(result.sources),
                **{stage: latency.get(stage, 0.0) for stage in STAGES},
            }
        )
        print(f"[{i:03d}/{n}] total={latency['total_ms']:.1f}ms generation={latency['generation_ms']:.1f}ms")

    reports = {name: summarize(name, values) for name, values in buckets.items()}
    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "request_count": n,
        "unique_query_count": unique_queries,
        "excluded_invalid_query_count": excluded_invalid_queries,
        "warmup_queries": warmup,
        "model_initialization_ms": model_initialization_ms,
        "parallel_retrieval": settings.parallel_retrieval,
        "response_cache_enabled": False,
        "index_chunk_count": len(pipeline.hybrid.bm25),
        "embedding_model": settings.embedding_model,
        "reranker_model": settings.reranker_model,
        "elevenlabs_model": settings.elevenlabs_model,
        "fusion_method": settings.fusion_method,
        "refused_count": refused,
    }
    output_dir = settings.benchmark_output_dir / "latency"
    _write_reports(output_dir, metadata=metadata, reports=reports, raw_rows=raw_rows)
    print(f"Wrote JSON/CSV/Markdown reports to {output_dir}")
    log.info("Benchmark complete", extra={"n": n, "output_dir": str(output_dir)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
