from __future__ import annotations

from rag.reranking.base import Reranker
from rag.retrieval.bm25 import tokenize
from rag.retrieval.types import RetrievalResult


class LexicalLightReranker(Reranker):
    """Deterministic CPU reranker: lexical overlap + incoming retrieval score."""

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        top_k: int,
    ) -> list[RetrievalResult]:
        query_tokens = set(tokenize(query))
        scored: list[RetrievalResult] = []
        for candidate in candidates:
            cand_tokens = set(tokenize(candidate.text))
            if query_tokens and cand_tokens:
                overlap = len(query_tokens & cand_tokens) / float(len(query_tokens))
            else:
                overlap = 0.0
            # Prefer lexical evidence while still respecting fused/retrieval scores.
            combined = (0.35 * float(candidate.score)) + (0.65 * overlap)
            metadata = dict(candidate.metadata)
            metadata["light_rerank_overlap"] = overlap
            metadata["light_rerank_score"] = combined
            scored.append(
                RetrievalResult(
                    text=candidate.text,
                    score=combined,
                    document_id=candidate.document_id,
                    chunk_id=candidate.chunk_id,
                    metadata=metadata,
                )
            )
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:top_k]
