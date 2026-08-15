from __future__ import annotations

from abc import ABC, abstractmethod

from rag.retrieval.types import RetrievalResult


class Reranker(ABC):
    @abstractmethod
    def rerank(self, query: str, candidates: list[RetrievalResult], top_k: int) -> list[RetrievalResult]:
        ...
