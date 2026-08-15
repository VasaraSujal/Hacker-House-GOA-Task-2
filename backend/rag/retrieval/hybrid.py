from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from rag.retrieval.bm25 import BM25Index
from rag.retrieval.dense import DenseRetriever
from rag.retrieval.fusion import reciprocal_rank_fusion, weighted_fusion
from rag.retrieval.types import RetrievalResult


@dataclass(slots=True)
class HybridSearchOutput:
    results: list[RetrievalResult]
    dense: list[RetrievalResult]
    bm25: list[RetrievalResult]
    embedding_ms: float
    dense_retrieval_ms: float
    bm25_ms: float
    retrieval_wall_ms: float
    fusion_ms: float


class HybridRetriever:
    def __init__(
        self,
        dense: DenseRetriever,
        bm25: BM25Index,
        *,
        fusion_method: str = "rrf",
        rrf_k: int = 60,
        dense_weight: float = 0.6,
        bm25_weight: float = 0.4,
        parallel: bool = True,
    ) -> None:
        self.dense = dense
        self.bm25 = bm25
        self.fusion_method = fusion_method
        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight
        self.parallel = parallel
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="retrieval")

    def _bm25_search(self, query: str, top_k: int) -> tuple[list[RetrievalResult], float]:
        started = time.perf_counter()
        hits = self.bm25.search(query, top_k=top_k)
        return hits, (time.perf_counter() - started) * 1000

    def search(
        self,
        query: str,
        *,
        dense_top_k: int = 20,
        bm25_top_k: int = 20,
        hybrid_top_k: int = 20,
    ) -> HybridSearchOutput:
        retrieval_started = time.perf_counter()
        if self.parallel:
            dense_future = self._executor.submit(self.dense.search, query, dense_top_k)
            bm25_future = self._executor.submit(self._bm25_search, query, bm25_top_k)
            dense_out = dense_future.result()
            bm25_hits, bm25_ms = bm25_future.result()
        else:
            dense_out = self.dense.search(query, top_k=dense_top_k)
            bm25_hits, bm25_ms = self._bm25_search(query, bm25_top_k)
        retrieval_wall_ms = (time.perf_counter() - retrieval_started) * 1000
        t1 = time.perf_counter()
        if self.fusion_method == "weighted":
            fused = weighted_fusion(
                dense_out.results,
                bm25_hits,
                dense_weight=self.dense_weight,
                sparse_weight=self.bm25_weight,
                top_k=hybrid_top_k,
            )
        else:
            fused = reciprocal_rank_fusion(
                [dense_out.results, bm25_hits],
                k=self.rrf_k,
                top_k=hybrid_top_k,
            )
        fusion_ms = (time.perf_counter() - t1) * 1000
        return HybridSearchOutput(
            results=fused,
            dense=dense_out.results,
            bm25=bm25_hits,
            embedding_ms=dense_out.embedding_ms,
            dense_retrieval_ms=dense_out.search_ms,
            bm25_ms=bm25_ms,
            retrieval_wall_ms=retrieval_wall_ms,
            fusion_ms=fusion_ms,
        )
