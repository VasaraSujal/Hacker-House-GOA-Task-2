from __future__ import annotations

from collections import defaultdict

from rag.retrieval.types import RetrievalResult


def _minmax(results: list[RetrievalResult]) -> dict[str, float]:
    if not results:
        return {}
    scores = [r.score for r in results]
    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-12:
        return {r.chunk_id: 1.0 for r in results}
    return {r.chunk_id: (r.score - lo) / (hi - lo) for r in results}


def reciprocal_rank_fusion(
    ranked_lists: list[list[RetrievalResult]],
    *,
    k: int = 60,
    top_k: int = 20,
) -> list[RetrievalResult]:
    """RRF: score = sum 1 / (k + rank). Does not require calibrated raw scores."""
    fused: dict[str, float] = defaultdict(float)
    docs: dict[str, RetrievalResult] = {}
    for results in ranked_lists:
        for rank, result in enumerate(results, start=1):
            fused[result.chunk_id] += 1.0 / (k + rank)
            docs[result.chunk_id] = result
    ordered = sorted(fused.items(), key=lambda item: item[1], reverse=True)[:top_k]
    out: list[RetrievalResult] = []
    for chunk_id, score in ordered:
        src = docs[chunk_id]
        out.append(
            RetrievalResult(
                text=src.text,
                score=float(score),
                document_id=src.document_id,
                chunk_id=src.chunk_id,
                metadata={**src.metadata, "fusion": "rrf"},
            )
        )
    return out


def weighted_fusion(
    dense: list[RetrievalResult],
    sparse: list[RetrievalResult],
    *,
    dense_weight: float = 0.6,
    sparse_weight: float = 0.4,
    top_k: int = 20,
) -> list[RetrievalResult]:
    dense_norm = _minmax(dense)
    sparse_norm = _minmax(sparse)
    docs: dict[str, RetrievalResult] = {r.chunk_id: r for r in dense}
    docs.update({r.chunk_id: r for r in sparse})
    fused: dict[str, float] = {}
    for chunk_id in docs:
        fused[chunk_id] = dense_weight * dense_norm.get(chunk_id, 0.0) + sparse_weight * sparse_norm.get(chunk_id, 0.0)
    ordered = sorted(fused.items(), key=lambda item: item[1], reverse=True)[:top_k]
    out: list[RetrievalResult] = []
    for chunk_id, score in ordered:
        src = docs[chunk_id]
        out.append(
            RetrievalResult(
                text=src.text,
                score=float(score),
                document_id=src.document_id,
                chunk_id=src.chunk_id,
                metadata={**src.metadata, "fusion": "weighted"},
            )
        )
    return out
