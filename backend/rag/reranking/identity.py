from __future__ import annotations

from rag.reranking.base import Reranker
from rag.retrieval.types import RetrievalResult


class IdentityReranker(Reranker):
    """Zero-cost reranker used when latency policy disables cross-encoding."""

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        top_k: int,
    ) -> list[RetrievalResult]:
        del query
        return candidates[:top_k]
