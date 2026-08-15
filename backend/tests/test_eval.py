from evaluation.latency_eval import percentile, summarize
from evaluation.retrieval_eval import mrr, ndcg_at_k, recall_at_k, score_ranking


def test_recall_mrr_ndcg() -> None:
    relevant = {"a", "b"}
    retrieved = ["x", "a", "b"]
    assert recall_at_k(relevant, retrieved, 5) == 1.0
    assert mrr(relevant, retrieved) == 0.5
    assert ndcg_at_k(relevant, retrieved, 10) > 0


def test_ndcg_penalizes_incomplete_recall() -> None:
    relevant = {"a", "b", "c"}
    retrieved = ["a"]
    # Old ideal-from-retrieved logic wrongly scored this as 1.0.
    assert ndcg_at_k(relevant, retrieved, 10) < 1.0


def test_score_ranking_keys() -> None:
    row = score_ranking({"a"}, ["a", "b"])
    assert set(row) == {"recall@5", "recall@10", "precision@5", "mrr", "ndcg@10"}


def test_percentiles_not_fabricated() -> None:
    values = [10.0, 20.0, 30.0, 40.0, 100.0]
    assert percentile(values, 50) == 30.0
    report = summarize("total_ms", values)
    assert report.p50_ms == 30.0
    assert report.p100_ms == 100.0
    assert report.min_ms == 10.0
    assert report.max_ms == 100.0
    assert report.p90_ms >= report.p70_ms
    assert report.n == 5
