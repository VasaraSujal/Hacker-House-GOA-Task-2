from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def recall_at_k(relevant: set[str], retrieved: list[str], k: int) -> float:
    if not relevant:
        return 0.0
    hit = len(relevant.intersection(retrieved[:k]))
    return hit / len(relevant)


def precision_at_k(relevant: set[str], retrieved: list[str], k: int) -> float:
    if k <= 0:
        return 0.0
    hit = len(relevant.intersection(retrieved[:k]))
    return hit / k


def mrr(relevant: set[str], retrieved: list[str]) -> float:
    for i, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1.0 / i
    return 0.0


def dcg(relevances: list[float]) -> float:
    if not relevances:
        return 0.0
    ranks = np.arange(1, len(relevances) + 1)
    return float(np.sum((np.power(2, relevances) - 1) / np.log2(ranks + 1)))


def ndcg_at_k(relevant: set[str], retrieved: list[str], k: int) -> float:
    """Binary nDCG@k with ideal DCG over the full relevant set, not only retrieved hits."""
    gains = [1.0 if doc_id in relevant else 0.0 for doc_id in retrieved[:k]]
    # Ideal ranking places every known relevant document first (capped at k).
    # Building the ideal from retrieved gains alone would score incomplete recall as 1.0.
    ideal = [1.0] * min(len(relevant), k)
    ideal_dcg = dcg(ideal)
    if ideal_dcg == 0:
        return 0.0
    return dcg(gains) / ideal_dcg


@dataclass
class RetrievalMetrics:
    recall_at_5: float
    recall_at_10: float
    precision_at_5: float
    mrr: float
    ndcg_at_10: float
    n_queries: int


def aggregate_metrics(per_query: list[dict[str, float]]) -> RetrievalMetrics:
    if not per_query:
        return RetrievalMetrics(0, 0, 0, 0, 0, 0)
    n = len(per_query)
    avg = lambda key: float(np.mean([row[key] for row in per_query]))
    return RetrievalMetrics(
        recall_at_5=avg("recall@5"),
        recall_at_10=avg("recall@10"),
        precision_at_5=avg("precision@5"),
        mrr=avg("mrr"),
        ndcg_at_10=avg("ndcg@10"),
        n_queries=n,
    )


def score_ranking(relevant_ids: set[str], ranked_ids: list[str]) -> dict[str, float]:
    return {
        "recall@5": recall_at_k(relevant_ids, ranked_ids, 5),
        "recall@10": recall_at_k(relevant_ids, ranked_ids, 10),
        "precision@5": precision_at_k(relevant_ids, ranked_ids, 5),
        "mrr": mrr(relevant_ids, ranked_ids),
        "ndcg@10": ndcg_at_k(relevant_ids, ranked_ids, 10),
    }
