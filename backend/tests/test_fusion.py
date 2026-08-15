from rag.retrieval.fusion import reciprocal_rank_fusion, weighted_fusion
from rag.retrieval.types import RetrievalResult


def _r(chunk_id: str, score: float, text: str = "t") -> RetrievalResult:
    return RetrievalResult(text=text, score=score, document_id=chunk_id, chunk_id=chunk_id)


def test_rrf_prefers_items_high_in_both_lists() -> None:
    dense = [_r("a", 0.9), _r("b", 0.8), _r("c", 0.1)]
    bm25 = [_r("c", 10), _r("a", 9), _r("d", 1)]
    fused = reciprocal_rank_fusion([dense, bm25], k=60, top_k=3)
    ids = [r.chunk_id for r in fused]
    assert "a" in ids
    assert fused[0].chunk_id in {"a", "c"}
    assert len(fused) == 3


def test_weighted_fusion_not_concatenation() -> None:
    dense = [_r("only_dense", 1.0), _r("both", 0.5)]
    sparse = [_r("only_bm25", 100.0), _r("both", 50.0)]
    fused = weighted_fusion(dense, sparse, dense_weight=0.5, sparse_weight=0.5, top_k=3)
    ids = {r.chunk_id for r in fused}
    assert ids == {"only_dense", "both", "only_bm25"}
    assert all(r.metadata.get("fusion") == "weighted" for r in fused)
