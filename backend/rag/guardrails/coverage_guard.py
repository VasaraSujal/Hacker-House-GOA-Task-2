from __future__ import annotations

import re
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

_NON_LATIN_NON_DEVA = re.compile(
    r"[\u0A80-\u0AFF\u0980-\u09FF\u0B80-\u0BFF\u0C00-\u0C7F\u0C80-\u0CFF\u0D00-\u0D7F\u0A00-\u0A7F\u0B00-\u0B7F\u0600-\u06FF]"
)


@dataclass(slots=True)
class CoverageResult:
    ok: bool
    best_overlap: float
    reason: str | None = None


class LexicalCoverageGuard:
    """Require substantive query↔evidence token overlap before answering.

    Used in extractive Free mode to refuse live-event / off-topic queries that
    may still retrieve weakly related passages via dense scores alone.
    When a query is in a non-Latin / non-Devanagari script (e.g. Gujarati) while
    evidence is in English/Hindi, character lexical overlap cannot occur; in this
    cross-script scenario, high-confidence dense semantic relevance is verified.
    """

    def __init__(
        self,
        min_overlap: float = 0.34,
        min_content_tokens: int = 1,
        min_cross_script_dense_score: float = 0.52,
    ) -> None:
        self.min_overlap = min_overlap
        self.min_content_tokens = min_content_tokens
        self.min_cross_script_dense_score = min_cross_script_dense_score

    def check(
        self,
        query: str,
        results: list[RetrievalResult],
        dense_results: list[RetrievalResult] | None = None,
    ) -> CoverageResult:
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
        if best >= self.min_overlap:
            return CoverageResult(True, best)

        # Cross-script check: when query is in an Indic/Arabic script not matching English/Devanagari
        if _NON_LATIN_NON_DEVA.search(query):
            pool = dense_results if dense_results is not None else results
            max_dense = max((r.score for r in pool[:3]), default=0.0)
            if max_dense >= self.min_cross_script_dense_score:
                return CoverageResult(True, best, "cross_script_semantic_match")
            return CoverageResult(
                False,
                best,
                f"insufficient cross-script dense confidence ({max_dense:.4f} < {self.min_cross_script_dense_score})",
            )

        return CoverageResult(False, best, "insufficient lexical coverage against retrieved evidence")

