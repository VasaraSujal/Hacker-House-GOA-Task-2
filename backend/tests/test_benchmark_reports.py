from evaluation.latency_eval import summarize
from scripts.benchmark import _write_reports


def test_benchmark_writer_creates_all_formats(tmp_path) -> None:
    reports = {
        "rag_core_ms": summarize("rag_core_ms", [10.0, 20.0]),
        "generation_ms": summarize("generation_ms", [100.0, 200.0]),
        "total_ms": summarize("total_ms", [110.0, 220.0]),
        "component_sum_ms": summarize("component_sum_ms", [110.0, 220.0]),
        "unaccounted_ms": summarize("unaccounted_ms", [0.0, 0.0]),
    }
    metadata = {
        "timestamp_utc": "2026-08-15T00:00:00+00:00",
        "request_count": 2,
        "unique_query_count": 2,
        "excluded_invalid_query_count": 0,
        "index_chunk_count": 100,
        "parallel_retrieval": True,
        "model_initialization_ms": 5.0,
        "refused_count": 0,
    }
    _write_reports(tmp_path, metadata=metadata, reports=reports, raw_rows=[])
    assert (tmp_path / "latency_report.json").exists()
    assert (tmp_path / "latency_report.csv").exists()
    assert (tmp_path / "latency_report.md").exists()
