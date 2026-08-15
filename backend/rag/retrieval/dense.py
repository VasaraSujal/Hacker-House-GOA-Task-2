from __future__ import annotations

import time
from dataclasses import dataclass

from rag.embeddings.base import EmbeddingProvider
from rag.retrieval.qdrant_store import QdrantStore
from rag.retrieval.types import RetrievalResult


@dataclass(slots=True)
class DenseSearchOutput:
    results: list[RetrievalResult]
    embedding_ms: float
    search_ms: float


class DenseRetriever:
    def __init__(self, embeddings: EmbeddingProvider, store: QdrantStore) -> None:
        self.embeddings = embeddings
        self.store = store

    def search(self, query: str, top_k: int = 20) -> DenseSearchOutput:
        t0 = time.perf_counter()
        vector = self.embeddings.embed_query(query)
        embedding_ms = (time.perf_counter() - t0) * 1000
        t1 = time.perf_counter()
        results = self.store.search(vector, top_k=top_k)
        search_ms = (time.perf_counter() - t1) * 1000
        return DenseSearchOutput(results=results, embedding_ms=embedding_ms, search_ms=search_ms)
