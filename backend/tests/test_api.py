from fastapi.testclient import TestClient

from app.main import create_app
from app.core.exceptions import GenerationError, STTError
from rag.pipeline import RAGPipeline
from rag.retrieval.bm25 import BM25Index
from rag.retrieval.dense import DenseRetriever
from rag.retrieval.hybrid import HybridRetriever
from rag.retrieval.types import RetrievalResult
from tests.conftest import FakeEmbeddings, FakeLLM, FakeReranker, FakeSTT, FakeStore
from app.core.config import Settings


def _client() -> TestClient:
    app = create_app(load_pipeline=False)
    hit = RetrievalResult("Paris is the capital of France.", 0.9, "d1", "c1")
    settings = Settings(elevenlabs_api_key="test", relevance_min_score=0.01)
    dense = DenseRetriever(FakeEmbeddings(), FakeStore(hits=[hit]))
    pipeline = RAGPipeline(
        settings=settings,
        hybrid=HybridRetriever(dense, BM25Index()),
        reranker=FakeReranker(),
        llm=FakeLLM("Paris is the capital of France."),
    )
    app.state.pipeline = pipeline
    app.state.stt = FakeSTT()
    return TestClient(app)


def test_health() -> None:
    client = _client()
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] in {"ok", "degraded"}
    assert "qdrant" in body


def test_cors_allows_frontend_development_origin() -> None:
    client = _client()
    response = client.options(
        "/api/voice/query",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_rag_query() -> None:
    client = _client()
    response = client.post("/api/rag/query", json={"query": "What is the capital of France?"})
    assert response.status_code == 200
    body = response.json()
    assert body["query"]
    assert body["answer"]
    assert "latency" in body
    assert "request_parsing_ms" in body["latency"]
    assert "query_processing_ms" in body["latency"]
    assert "context_building_ms" in body["latency"]
    assert "grounding_ms" in body["latency"]
    assert "rag_core_ms" in body["latency"]
    assert "grounded" in body
    assert isinstance(body["sources"], list)


def test_rag_query_empty_rejected() -> None:
    client = _client()
    response = client.post("/api/rag/query", json={"query": "   "})
    assert response.status_code in {400, 422}


def test_rag_query_rejects_overlong_payload() -> None:
    client = _client()
    response = client.post("/api/rag/query", json={"query": "x" * 2001})
    assert response.status_code == 422


def test_voice_query_transcribes_then_reuses_rag_pipeline() -> None:
    client = _client()
    response = client.post(
        "/api/voice/query",
        files={"audio": ("question.wav", b"RIFF-test-audio", "audio/wav")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["transcript"] == "What is the capital of France?"
    assert body["grounded"] is True
    assert body["sources"]
    assert body["latency"]["stt_ms"] == 2.0
    assert "rag_ms" in body["latency"]


def test_voice_query_accepts_browser_webm_codec_parameter() -> None:
    client = _client()
    response = client.post(
        "/api/voice/query",
        files={"audio": ("question.webm", b"browser-webm-audio", "audio/webm;codecs=opus")},
    )
    assert response.status_code == 200


def test_voice_query_rejects_unsupported_format() -> None:
    client = _client()
    response = client.post(
        "/api/voice/query",
        files={"audio": ("question.txt", b"not audio", "text/plain")},
    )
    assert response.status_code == 400


def test_voice_query_empty_transcript_does_not_call_rag() -> None:
    client = _client()
    client.app.state.stt = FakeSTT("")
    response = client.post(
        "/api/voice/query",
        files={"audio": ("question.wav", b"RIFF-test-audio", "audio/wav")},
    )
    assert response.status_code == 400


def test_voice_query_stt_failure_is_controlled() -> None:
    client = _client()

    class FailingSTT:
        def transcribe(self, *args, **kwargs):
            raise STTError("provider unavailable", status_code=502)

    client.app.state.stt = FailingSTT()
    response = client.post(
        "/api/voice/query",
        files={"audio": ("question.wav", b"RIFF-test-audio", "audio/wav")},
    )
    assert response.status_code == 502


def test_voice_query_rag_failure_is_controlled(monkeypatch) -> None:
    client = _client()
    monkeypatch.setattr(
        client.app.state.pipeline,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(GenerationError("generation unavailable")),
    )
    response = client.post(
        "/api/voice/query",
        files={"audio": ("question.wav", b"RIFF-test-audio", "audio/wav")},
    )
    assert response.status_code == 502


def test_voice_query_rate_limit_returns_429() -> None:
    from app.core.rate_limit import SlidingWindowRateLimiter

    client = _client()
    client.app.state.voice_limiter = SlidingWindowRateLimiter(limit=1, window_seconds=60.0)
    first = client.post(
        "/api/voice/query",
        files={"audio": ("question.wav", b"RIFF-test-audio", "audio/wav")},
    )
    second = client.post(
        "/api/voice/query",
        files={"audio": ("question.wav", b"RIFF-test-audio", "audio/wav")},
    )
    assert first.status_code == 200
    assert second.status_code == 429
    assert second.json()["detail"]["code"] == "voice_rate_limited"
