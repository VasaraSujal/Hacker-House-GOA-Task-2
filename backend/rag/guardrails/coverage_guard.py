from __future__ import annotations

from dataclasses import dataclass

from rag.retrieval.bm25 import tokenize
from rag.retrieval.types import RetrievalResult

_STOP = {
    "a",
    "an",
    "the",
    "is",
    "are",
    "was",
    "were",
    "be",
    "to",
    "of",
    "in",
    "on",
    "for",
    "and",
    "or",
    "who",
    "what",
    "when",
    "where",
    "why",
    "how",
    "did",
    "does",
    "do",
    "won",
    "yesterday",
    "today",
    "tomorrow",
    "match",
    "game",
}


@dataclass(slots=True)
class CoverageResult:
    ok: bool
    best_overlap: float
    reason: str | None = None


class LexicalCoverageGuard:
    """Require substantive query↔evidence token overlap before answering.

    Used in extractive Free mode to refuse live-event / off-topic queries that
    may still retrieve weakly related passages via dense scores alone.
    """

    def __init__(self, min_overlap: float = 0.34, min_content_tokens: int = 1) -> None:
        self.min_overlap = min_overlap
        self.min_content_tokens = min_content_tokens

    def check(self, query: str, results: list[RetrievalResult]) -> CoverageResult:
        query_tokens = {t for t in tokenize(query) if t not in _STOP and len(t) > 2}
        if len(query_tokens) < self.min_content_tokens:
            return CoverageResult(False, 0.0, "query has insufficient content tokens")
        if not results:
            return CoverageResult(False, 0.0, "no retrieval results")
        best = 0.0
        for result in results[:8]:
            doc_tokens = set(tokenize(result.text))
            if not doc_tokens:
                continue
            overlap = len(query_tokens & doc_tokens) / float(len(query_tokens))
            if overlap > best:
                best = overlap
        if best < self.min_overlap:
            return CoverageResult(False, best, "insufficient lexical coverage against retrieved evidence")
        return CoverageResult(True, best)
