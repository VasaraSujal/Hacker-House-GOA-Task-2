from rag.retrieval.dense import DenseRetriever
from rag.retrieval.bm25 import BM25Index
from rag.retrieval.hybrid import HybridRetriever
from rag.retrieval.types import RetrievalResult
from tests.conftest import FakeEmbeddings, FakeStore


def test_dense_retriever_returns_store_hits() -> None:
    hit = RetrievalResult("Paris is the capital of France.", 0.87, "doc-1", "chunk-1", {"language": "en"})
    retriever = DenseRetriever(FakeEmbeddings(), FakeStore(hits=[hit]))
    out = retriever.search("capital of France", top_k=5)
    assert out.results[0].document_id == "doc-1"
    assert out.results[0].chunk_id == "chunk-1"
    assert out.embedding_ms >= 0
    assert out.search_ms >= 0


def test_parallel_hybrid_reports_wall_latency() -> None:
    hit = RetrievalResult("Paris is the capital of France.", 0.87, "doc-1", "chunk-1")
    dense = DenseRetriever(FakeEmbeddings(), FakeStore(hits=[hit]))
    hybrid = HybridRetriever(dense, BM25Index(), parallel=True)
    out = hybrid.search("capital of France", dense_top_k=5, bm25_top_k=5, hybrid_top_k=5)
    assert out.results
    assert out.retrieval_wall_ms >= 0
    assert out.embedding_ms >= 0
    assert out.dense_retrieval_ms >= 0
    assert out.bm25_ms >= 0
