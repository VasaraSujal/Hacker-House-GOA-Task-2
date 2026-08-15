from __future__ import annotations

from dataclasses import dataclass

from rag.generation.prompts import REFUSAL_MESSAGE
from rag.retrieval.types import RetrievalResult


@dataclass(slots=True)
class RelevanceDecision:
    ok: bool
    reason: str
    max_score: float
    n_results: int


class RelevanceGuard:
    def __init__(self, min_results: int = 1, min_score: float = 0.05) -> None:
        self.min_results = min_results
        self.min_score = min_score
        self.refusal_message = REFUSAL_MESSAGE

    def check(self, results: list[RetrievalResult]) -> RelevanceDecision:
        if not results:
            return RelevanceDecision(False, "no_results", 0.0, 0)
        max_score = max(r.score for r in results)
        if len(results) < self.min_results:
            return RelevanceDecision(False, "too_few_results", max_score, len(results))
        if max_score < self.min_score:
            return RelevanceDecision(False, "low_score", max_score, len(results))
        return RelevanceDecision(True, "ok", max_score, len(results))
