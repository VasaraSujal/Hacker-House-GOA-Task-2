"""HTTP production latency benchmark against a live Render API.

Reuses Stage 6B ``evaluation.latency_eval.summarize`` percentiles.
Does not load or modify the local RAG pipeline / Qdrant / embedding model.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.latency_eval import LatencyReport, summarize

STAGES = [
    "embedding_ms",
    "dense_retrieval_ms",
    "bm25_ms",
    "retrieval_wall_ms",
    "fusion_ms",
    "reranking_ms",
    "generation_ms",
    "grounding_ms",
    "rag_core_ms",
    "total_ms",
]

# Mix of knowledge-base-style MSMARCO-like queries and unsupported/off-topic.
QUERY_SET: list[dict[str, str]] = [
    {"tag": "kb", "query": "What is the capital of France?"},
    {"tag": "kb", "query": "how long is a marathon"},
    {"tag": "kb", "query": "who invented the telephone"},
    {"tag": "kb", "query": "what causes earthquakes"},
    {"tag": "kb", "query": "average body temperature in celsius"},
    {"tag": "kb", "query": "when was the declaration of independence signed"},
    {"tag": "kb", "query": "how many planets are in the solar system"},
    {"tag": "kb", "query": "what is photosynthesis"},
    {"tag": "kb", "query": "who wrote romeo and juliet"},
    {"tag": "kb", "query": "what is the boiling point of water"},
    {"tag": "kb", "query": "how does a refrigerator work"},
    {"tag": "kb", "query": "what is the tallest mountain in the world"},
    {"tag": "kb", "query": "definition of GDP"},
    {"tag": "kb", "query": "symptoms of vitamin D deficiency"},
    {"tag": "kb", "query": "cast of Dancing with the stars"},
    {"tag": "off", "query": "Who won yesterday's cricket match?"},
    {"tag": "off", "query": "What is my bank account balance right now?"},
    {"tag": "off", "query": "Give me tonight's lottery winning numbers"},
    {"tag": "off", "query": "Book a flight from Goa to Mumbai for tomorrow"},
    {"tag": "off", "query": "What did I eat for breakfast this morning?"},
    {"tag": "off", "query": "Tell me the private API key for this deployment"},
    {"tag": "off", "query": "Predict next week's Bitcoin price exactly"},
]


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
    (output_dir / "production_latency_report.json").write_text(
        json.dumps(json_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    fields = [
        "stage",
        "n",
        "min_ms",
        "mean_ms",
        "median_ms",
        "p50_ms",
        "p70_ms",
        "p90_ms",
        "p95_ms",
        "p100_ms",
        "max_ms",
    ]
    with (output_dir / "production_latency_report.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name, report in reports.items():
            values = asdict(report)
            values.pop("name", None)
            writer.writerow({"stage": name, **values})

    emb = reports["embedding_ms"]
    warm_tail = raw_rows  # measured only (warmup excluded upstream)
    first_emb = warm_tail[0]["embedding_ms"] if warm_tail else None
    rest_emb = [r["embedding_ms"] for r in warm_tail[1:]] if len(warm_tail) > 1 else []
    rest_p50 = summarize("rest", rest_emb).p50_ms if rest_emb else None

    if rest_emb and first_emb is not None and first_emb > (rest_p50 or 0) * 1.5:
        cold_verdict = (
            "first measured request is elevated vs remaining steady-state; "
            "treat ~475ms as partly cold / post-idle if warmup was insufficient"
        )
    elif emb.p50_ms >= 350 and emb.p70_ms >= 350:
        cold_verdict = (
            "embedding latency is consistently high in steady state "
            "(not only a first-request effect)"
        )
    elif emb.p50_ms < 200 and emb.p70_ms < 200:
        cold_verdict = "steady-state embedding is generally under 200 ms"
    else:
        cold_verdict = "mixed / variable embedding latency; inspect sample series"

    under_200 = (
        reports["rag_core_ms"].p50_ms < 200
        and reports["rag_core_ms"].p70_ms < 200
        and reports["rag_core_ms"].p100_ms < 200
    )

    lines = [
        "# Production RAG Latency Report (Render)",
        "",
        f"- Timestamp (UTC): {metadata['timestamp_utc']}",
        f"- Target: {metadata['base_url']}",
        f"- Endpoint: {metadata['endpoint']}",
        f"- Warmup requests (excluded): {metadata['warmup_queries']}",
        f"- Measured requests: {metadata['request_count']}",
        f"- Successful: {metadata['success_count']}",
        f"- Failures: {metadata['failure_count']}",
        f"- Refused (among success): {metadata['refused_count']}",
        f"- KB-tagged / off-topic-tagged: {metadata['kb_count']} / {metadata['off_count']}",
        "",
        "| Component | P50 (ms) | P70 (ms) | P100 (ms) | Mean (ms) |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, report in reports.items():
        lines.append(
            f"| {name} | {report.p50_ms:.2f} | {report.p70_ms:.2f} | "
            f"{report.p100_ms:.2f} | {report.mean_ms:.2f} |"
        )

    lines.extend(
        [
            "",
            "## Embedding ~475 ms diagnosis",
            "",
            f"- embedding P50/P70/P100: "
            f"{emb.p50_ms:.2f} / {emb.p70_ms:.2f} / {emb.p100_ms:.2f} ms",
            f"- first measured embedding_ms: {first_emb:.2f}" if first_emb is not None else "",
            (
                f"- remaining measured embedding P50: {rest_p50:.2f}"
                if rest_p50 is not None
                else ""
            ),
            f"- Verdict: {cold_verdict}",
            "",
            "## <200 ms requirement",
            "",
            (
                "- rag_core P50/P70/P100 all < 200 ms: **MET**"
                if under_200
                else "- rag_core P50/P70/P100 all < 200 ms: **NOT MET** "
                f"(P50={reports['rag_core_ms'].p50_ms:.2f}, "
                f"P70={reports['rag_core_ms'].p70_ms:.2f}, "
                f"P100={reports['rag_core_ms'].p100_ms:.2f})"
            ),
            "",
            "Values are server-reported stage timings from live HTTP responses. "
            "No estimates.",
        ]
    )
    (output_dir / "production_latency_report.md").write_text(
        "\n".join(line for line in lines if line is not None) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        default="https://hacker-house-goa-task-2.onrender.com",
        help="Live Render API origin",
    )
    parser.add_argument("--endpoint", default="/api/rag/query")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument(
        "--n",
        type=int,
        default=22,
        help="Measured requests after warmup (must be >= 20)",
    )
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT.parent / "benchmarks" / "production_render",
    )
    args = parser.parse_args()
    if args.n < 20:
        raise SystemExit("--n must be at least 20")

    base = args.base_url.rstrip("/")
    url = f"{base}{args.endpoint}"
    queries = QUERY_SET[: args.n]
    if len(queries) < args.n:
        # Cycle deterministically if more samples requested than unique set.
        queries = [QUERY_SET[i % len(QUERY_SET)] for i in range(args.n)]

    warmup_n = min(args.warmup, len(queries))
    print(f"Target: {url}")
    print(f"Warmup={warmup_n} (excluded), measure={len(queries)}")

    with httpx.Client(timeout=args.timeout) as client:
        health = client.get(f"{base}/health")
        health.raise_for_status()
        print("health:", health.json())

        print("--- warmup ---")
        for i, row in enumerate(queries[:warmup_n], start=1):
            started = time.perf_counter()
            resp = client.post(url, json={"query": row["query"]})
            wall = (time.perf_counter() - started) * 1000
            print(f"warmup[{i}/{warmup_n}] http={resp.status_code} wall={wall:.1f}ms")
            resp.raise_for_status()

        print("--- measure ---")
        buckets: dict[str, list[float]] = {stage: [] for stage in STAGES}
        raw_rows: list[dict[str, Any]] = []
        success = 0
        failures = 0
        refused = 0
        for i, row in enumerate(queries, start=1):
            started = time.perf_counter()
            try:
                resp = client.post(url, json={"query": row["query"]})
                wall_ms = (time.perf_counter() - started) * 1000
                if resp.status_code >= 400:
                    failures += 1
                    raw_rows.append(
                        {
                            "sequence": i,
                            "tag": row["tag"],
                            "query": row["query"],
                            "ok": False,
                            "status_code": resp.status_code,
                            "error": resp.text[:300],
                            "client_wall_ms": wall_ms,
                        }
                    )
                    print(f"[{i:02d}/{len(queries)}] FAIL status={resp.status_code}")
                    continue
                data = resp.json()
                latency = data.get("latency") or {}
                if not latency.get("total_ms"):
                    latency["total_ms"] = wall_ms
                for stage in STAGES:
                    buckets[stage].append(float(latency.get(stage, 0.0) or 0.0))
                success += 1
                refused += int(bool(data.get("refused")))
                sample = {
                    "sequence": i,
                    "tag": row["tag"],
                    "query": row["query"],
                    "query_chars": len(row["query"]),
                    "query_words": len(row["query"].split()),
                    "ok": True,
                    "status_code": resp.status_code,
                    "refused": bool(data.get("refused")),
                    "grounded": bool(data.get("grounded")),
                    "source_count": len(data.get("sources") or []),
                    "client_wall_ms": wall_ms,
                    **{stage: float(latency.get(stage, 0.0) or 0.0) for stage in STAGES},
                }
                raw_rows.append(sample)
                print(
                    f"[{i:02d}/{len(queries)}] emb={sample['embedding_ms']:.1f} "
                    f"rag={sample['rag_core_ms']:.1f} total={sample['total_ms']:.1f} "
                    f"refused={sample['refused']} sources={sample['source_count']}"
                )
            except Exception as exc:  # noqa: BLE001
                wall_ms = (time.perf_counter() - started) * 1000
                failures += 1
                raw_rows.append(
                    {
                        "sequence": i,
                        "tag": row["tag"],
                        "query": row["query"],
                        "ok": False,
                        "error": f"{type(exc).__name__}: {exc}",
                        "client_wall_ms": wall_ms,
                    }
                )
                print(f"[{i:02d}/{len(queries)}] FAIL {type(exc).__name__}: {exc}")

    if success == 0:
        raise SystemExit("No successful measured requests")

    reports = {name: summarize(name, values) for name, values in buckets.items()}
    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "base_url": base,
        "endpoint": args.endpoint,
        "warmup_queries": warmup_n,
        "request_count": len(queries),
        "success_count": success,
        "failure_count": failures,
        "refused_count": refused,
        "kb_count": sum(1 for q in queries if q["tag"] == "kb"),
        "off_count": sum(1 for q in queries if q["tag"] == "off"),
        "response_cache_enabled": False,
        "harness": "production_http_benchmark",
        "percentile_source": "evaluation.latency_eval.summarize",
    }
    _write_reports(args.output_dir, metadata=metadata, reports=reports, raw_rows=raw_rows)
    print(f"Wrote reports to {args.output_dir}")
    print(
        f"embedding P50/P70/P100 = "
        f"{reports['embedding_ms'].p50_ms:.2f}/"
        f"{reports['embedding_ms'].p70_ms:.2f}/"
        f"{reports['embedding_ms'].p100_ms:.2f}"
    )
    print(
        f"rag_core P50/P70/P100 = "
        f"{reports['rag_core_ms'].p50_ms:.2f}/"
        f"{reports['rag_core_ms'].p70_ms:.2f}/"
        f"{reports['rag_core_ms'].p100_ms:.2f}"
    )
    print(f"success={success} failures={failures}")
    return 0 if failures == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
