from app.core.config import Settings
from rag.pipeline import RAGPipeline
from rag.retrieval.bm25 import BM25Index
from rag.retrieval.dense import DenseRetriever
from rag.retrieval.hybrid import HybridRetriever
from rag.retrieval.types import RetrievalResult
from tests.conftest import FakeEmbeddings, FakeLLM, FakeReranker, FakeStore


def _pipeline(llm_text: str, hits: list[RetrievalResult], min_score: float = 0.01) -> RAGPipeline:
    settings = Settings(
        elevenlabs_api_key="test",
        relevance_min_score=min_score,
        grounding_min_overlap=0.05,
        rerank_top_k=3,
        hybrid_top_k=5,
    )
    dense = DenseRetriever(FakeEmbeddings(), FakeStore(hits=hits))
    hybrid = HybridRetriever(dense, BM25Index())
    return RAGPipeline(
        settings=settings,
        hybrid=hybrid,
        reranker=FakeReranker(),
        llm=FakeLLM(llm_text),
    )


def test_pipeline_grounded_answer() -> None:
    hit = RetrievalResult(
        "Paris is the capital of France and sits on the Seine.",
        0.9,
        "doc-1",
        "chunk-1",
        {"language": "en"},
    )
    pipe = _pipeline("Paris is the capital of France.", [hit])
    result = pipe.run("What is the capital of France?")
    assert result.grounded
    assert not result.refused
    assert "Paris" in result.answer
    assert result.sources[0].document_id == "doc-1"
    assert result.latency.total_ms >= 0
    assert result.latency.query_processing_ms >= 0
    assert result.latency.retrieval_wall_ms >= 0
    assert result.latency.context_building_ms >= 0
    assert result.latency.grounding_ms >= 0
    assert result.latency.rag_core_ms >= 0
    assert result.latency.component_sum_ms <= result.latency.total_ms + 1.0
    assert result.latency.unaccounted_ms >= 0


def test_pipeline_refuses_when_retrieval_empty() -> None:
    pipe = _pipeline("should not be called", [])
    llm: FakeLLM = pipe.llm  # type: ignore[assignment]
    result = pipe.run("What is the capital of Atlantis?")
    assert result.refused
    assert result.grounded
    assert llm.calls == 0
    assert "couldn't find enough relevant information" in result.answer.lower()


def test_pipeline_rejects_empty_query() -> None:
    pipe = _pipeline("x", [])
    try:
        pipe.run("   ")
        assert False, "expected invalid query"
    except Exception as exc:
        assert "empty" in str(exc).lower() or "invalid" in str(exc).lower()
