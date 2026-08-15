from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest

from app.core.config import Settings
from app.models.schemas import LatencyBreakdown, RAGResponse
from rag.embeddings.base import EmbeddingProvider
from rag.generation.base import GenerationResult, LLMProvider
from rag.reranking.base import Reranker
from rag.retrieval.types import RetrievalResult
from rag.stt.base import Transcript


class FakeEmbeddings(EmbeddingProvider):
    def __init__(self, dim: int = 8) -> None:
        self._dim = dim

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return "fake"

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        rows = []
        for text in texts:
            rng = np.random.default_rng(abs(hash(text)) % (2**32))
            vec = rng.normal(size=self._dim).astype(np.float32)
            vec /= np.linalg.norm(vec) + 1e-8
            rows.append(vec)
        return np.vstack(rows) if rows else np.zeros((0, self._dim), dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]


class FakeLLM(LLMProvider):
    def __init__(self, text: str = "The capital mentioned in the context is Paris.") -> None:
        self.text = text
        self.calls = 0

    def generate(self, prompt: str, *, system_prompt: str | None = None) -> GenerationResult:
        self.calls += 1
        return GenerationResult(text=self.text, latency_ms=1.5, model="fake")


class FakeReranker(Reranker):
    def rerank(self, query: str, candidates: list[RetrievalResult], top_k: int) -> list[RetrievalResult]:
        return candidates[:top_k]


class FakeSTT:
    def __init__(self, text: str = "What is the capital of France?") -> None:
        self.text = text
        self.calls = 0

    def transcribe(self, audio_bytes: bytes, *, filename: str, content_type: str) -> Transcript:
        self.calls += 1
        return Transcript(text=self.text, language="en", latency_ms=2.0)


@dataclass
class FakeStore:
    hits: list[RetrievalResult] = field(default_factory=list)

    def search(self, vector, top_k: int = 20) -> list[RetrievalResult]:
        return self.hits[:top_k]

    def ping(self) -> bool:
        return True


@pytest.fixture
def settings() -> Settings:
    return Settings(
        elevenlabs_api_key="test-key",
        max_query_chars=200,
        relevance_min_score=0.01,
        grounding_min_overlap=0.05,
    )
