"""Measure the reranker and retrieval-concurrency optimization cycle."""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import get_settings
from app.dependencies import build_pipeline
from evaluation.latency_eval import summarize
from rag.generation.base import GenerationResult, LLMProvider


class RefusalBenchmarkLLM(LLMProvider):
    """Local deterministic provider; isolates RAG core without API generation."""

    def __init__(self, refusal_message: str) -> None:
        self.refusal_message = refusal_message

    def generate(self, prompt: str, *, system_prompt: str | None = None) -> GenerationResult:
        del prompt, system_prompt
        return GenerationResult(
            text=self.refusal_message,
            latency_ms=0.0,
            model="benchmark-no-generation",
        )


def _run_mode(pipeline, queries: list[str], *, parallel: bool) -> dict:
    pipeline.hybrid.parallel = parallel
    for query in queries[:3]:
        pipeline.run(query)
    core_values = []
    retrieval_values = []
    total_values = []
    for i, query in enumerate(queries, start=1):
        result = pipeline.run(query)
        core_values.append(result.latency.rag_core_ms)
        retrieval_values.append(result.latency.retrieval_wall_ms)
        total_values.append(result.latency.total_ms)
        print(f"{'parallel' if parallel else 'sequential'} [{i:03d}/{len(queries)}]")
    return {
        "rag_core": asdict(summarize("rag_core_ms", core_values)),
        "retrieval_wall": asdict(summarize("retrieval_wall_ms", retrieval_values)),
        "total_without_generation": asdict(summarize("total_ms", total_values)),
    }


def main() -> int:
    settings = get_settings()
    pipeline = build_pipeline(settings)
    pipeline.llm = RefusalBenchmarkLLM(settings.refusal_message)
    query_rows = [
        row
        for row in pipeline.hybrid.bm25.evaluation_queries()
        if 0 < len(row["query"]) <= settings.max_query_chars
    ][: settings.benchmark_queries]
    queries = [row["query"] for row in query_rows]
    if len(queries) < settings.benchmark_queries:
        raise RuntimeError("Not enough unique valid indexed queries for optimization benchmark")

    sequential = _run_mode(pipeline, queries, parallel=False)
    parallel = _run_mode(pipeline, queries, parallel=True)
    baseline_path = settings.benchmark_output_dir / "latency" / "latency_report.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline_stages = baseline["stages"]
    retrieval_path = settings.benchmark_output_dir / "retrieval" / "retrieval_report.json"
    retrieval_report = json.loads(retrieval_path.read_text(encoding="utf-8"))
    retrieval_systems = {
        row["pipeline"]: row for row in retrieval_report["systems"]
    }
    with_reranker = retrieval_systems["hybrid_rrf+reranker"]
    without_reranker = retrieval_systems["hybrid_rrf"]
    recommended_parallel = (
        parallel["retrieval_wall"]["p50_ms"]
        < sequential["retrieval_wall"]["p50_ms"]
    )
    report = {
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "query_count": len(queries),
            "generation": "disabled with deterministic local refusal provider",
            "reranker": "disabled via IdentityReranker",
            "baseline_source": str(baseline_path),
            "recommended_parallel_retrieval": recommended_parallel,
        },
        "before": {
            "rag_core": baseline_stages["rag_core_ms"],
            "retrieval_wall": baseline_stages["retrieval_wall_ms"],
            "reranking": baseline_stages["reranking_ms"],
            "generation": baseline_stages["generation_ms"],
            "total": baseline_stages["total_ms"],
        },
        "after_sequential": sequential,
        "after_parallel": parallel,
        "quality_tradeoff": {
            "with_reranker": {
                "mrr": with_reranker["mrr"],
                "ndcg_at_10": with_reranker["ndcg_at_10"],
            },
            "without_reranker_hybrid_rrf": {
                "mrr": without_reranker["mrr"],
                "ndcg_at_10": without_reranker["ndcg_at_10"],
            },
            "source": str(retrieval_path),
        },
    }
    output_dir = settings.benchmark_output_dir / "latency"
    (output_dir / "optimization_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    csv_rows = [
        {
            "configuration": "before_reranker_enabled_parallel",
            "rag_core_p50_ms": baseline_stages["rag_core_ms"]["p50_ms"],
            "retrieval_p50_ms": baseline_stages["retrieval_wall_ms"]["p50_ms"],
            "reranker_p50_ms": baseline_stages["reranking_ms"]["p50_ms"],
        },
        {
            "configuration": "after_reranker_disabled_sequential",
            "rag_core_p50_ms": sequential["rag_core"]["p50_ms"],
            "retrieval_p50_ms": sequential["retrieval_wall"]["p50_ms"],
            "reranker_p50_ms": 0.0,
        },
        {
            "configuration": "after_reranker_disabled_parallel",
            "rag_core_p50_ms": parallel["rag_core"]["p50_ms"],
            "retrieval_p50_ms": parallel["retrieval_wall"]["p50_ms"],
            "reranker_p50_ms": 0.0,
        },
    ]
    with (output_dir / "optimization_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)
    chosen = parallel if recommended_parallel else sequential
    chosen_name = "parallel" if recommended_parallel else "sequential"
    lines = [
        "# Measured Optimization Cycle",
        "",
        f"- Queries per configuration: {len(queries)}",
        "- Generation disabled in after-runs to isolate the RAG core.",
        "- Response cache disabled.",
        "",
        f"Before: reranker enabled, RAG core P50 **{baseline_stages['rag_core_ms']['p50_ms']:.2f} ms**.",
        f"After: reranker disabled, {chosen_name} retrieval, RAG core P50 "
        f"**{chosen['rag_core']['p50_ms']:.2f} ms**.",
        "",
        f"Sequential retrieval P50: **{sequential['retrieval_wall']['p50_ms']:.2f} ms**.",
        f"Parallel retrieval P50: **{parallel['retrieval_wall']['p50_ms']:.2f} ms**.",
        "",
        "Quality trade-off (50 labeled queries):",
        f"- RRF + reranker: MRR {with_reranker['mrr']:.4f}, "
        f"nDCG@10 {with_reranker['ndcg_at_10']:.4f}.",
        f"- RRF without reranker: MRR {without_reranker['mrr']:.4f}, "
        f"nDCG@10 {without_reranker['ndcg_at_10']:.4f}.",
        "",
        "ElevenLabs remains outside this optimization: its measured P50 is "
        f"**{baseline_stages['generation_ms']['p50_ms']:.2f} ms**.",
    ]
    (output_dir / "optimization_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Recommended retrieval mode: {chosen_name}")
    print(f"Wrote optimization reports to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
