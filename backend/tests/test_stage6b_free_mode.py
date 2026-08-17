from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.core.config import Settings
from app.dependencies import build_pipeline, validate_deployment_profile
from app.core.exceptions import EmbeddingError
from rag.generation.extractive import ExtractiveAnswerProvider
from rag.generation.prompts import REFUSAL_MESSAGE, build_user_prompt
from rag.reranking.lexical import LexicalLightReranker
from rag.retrieval.cloud_dense import CloudDenseRetriever
from rag.retrieval.types import RetrievalResult


def test_extractive_answer_uses_only_source_sentences() -> None:
    context = (
        "[Source 1]\nDocument ID: d1\nChunk ID: c1\n"
        "Text:\nA corporation is a company authorized to act as a single legal entity.\n\n"
        "[Source 2]\nDocument ID: d2\nChunk ID: c2\n"
        "Text:\nUnrelated weather systems move across the ocean daily."
    )
    prompt = build_user_prompt("What is a corporation?", context)
    result = ExtractiveAnswerProvider(max_sentences=2, min_overlap=0.05).generate(prompt)
    assert "corporation" in result.text.lower()
    assert "weather" not in result.text.lower()
    assert result.model == "extractive"


def test_extractive_answer_refuses_without_overlap() -> None:
    context = "[Source 1]\nDocument ID: d1\nChunk ID: c1\nText:\nBananas grow on trees in tropical climates."
    prompt = build_user_prompt("Who won yesterday's cricket match?", context)
    result = ExtractiveAnswerProvider(min_overlap=0.2).generate(prompt)
    assert result.text == REFUSAL_MESSAGE


def test_lexical_light_reranker_is_deterministic() -> None:
    candidates = [
        RetrievalResult("A corporation issues stock to shareholders.", 0.2, "d1", "c1"),
        RetrievalResult("Weather forecasts predict rain tomorrow.", 0.9, "d2", "c2"),
        RetrievalResult("A corporation is a legal person under company law.", 0.3, "d3", "c3"),
    ]
    reranker = LexicalLightReranker()
    first = reranker.rerank("What is a corporation?", candidates, top_k=2)
    second = reranker.rerank("What is a corporation?", candidates, top_k=2)
    assert [item.chunk_id for item in first] == [item.chunk_id for item in second]
    assert first[0].chunk_id in {"c1", "c3"}


def test_cloud_mode_requires_qdrant_credentials(tmp_path) -> None:
    settings = Settings(
        retrieval_mode="cloud_dense_sparse",
        answer_mode="extractive",
        qdrant_url="https://example.cloud.qdrant.io:6333",
        qdrant_api_key="",
        qdrant_collection="hh_goa_voice_rag_prod",
        bm25_index_path=tmp_path / "missing.pkl",
        app_env="development",
    )
    with pytest.raises(RuntimeError, match="QDRANT_API_KEY"):
        build_pipeline(settings)


def test_render_free_profile_rejects_local_retrieval() -> None:
    settings = Settings(
        deployment_profile="render_free",
        retrieval_mode="local",
        answer_mode="extractive",
    )
    with pytest.raises(RuntimeError, match="RETRIEVAL_MODE=cloud_dense_sparse"):
        validate_deployment_profile(settings)


def test_render_free_profile_rejects_generative_answer() -> None:
    settings = Settings(
        deployment_profile="render_free",
        retrieval_mode="cloud_dense_sparse",
        answer_mode="generative",
    )
    with pytest.raises(RuntimeError, match="ANSWER_MODE=extractive"):
        validate_deployment_profile(settings)


def test_local_embedding_provider_fails_before_loading_in_free_profile(monkeypatch) -> None:
    monkeypatch.setenv("DEPLOYMENT_PROFILE", "render_free")
    from rag.embeddings.local import LocalEmbeddingProvider

    with pytest.raises(EmbeddingError, match="attempted to initialize local embedding"):
        LocalEmbeddingProvider("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")


def test_build_pipeline_cloud_mode_uses_cloud_dense_and_extractive(tmp_path) -> None:
    # Minimal BM25 pickle via BM25Index API
    from rag.chunking.base import Chunk
    from rag.retrieval.bm25 import BM25Index

    bm25_path = tmp_path / "bm25.pkl"
    index = BM25Index()
    index.add_chunks(
        [
            Chunk(
                text="A corporation is a legal entity.",
                document_id="d1",
                chunk_id="c1",
                language="en",
                chunk_strategy="sentence",
                position=0,
            )
        ]
    )
    index.save(bm25_path)

    settings = Settings(
        deployment_profile="render_free",
        retrieval_mode="cloud_dense_sparse",
        answer_mode="extractive",
        qdrant_url="https://example.cloud.qdrant.io:6333",
        qdrant_api_key="test-key",
        qdrant_collection="hh_goa_voice_rag_prod",
        qdrant_inference_model="intfloat/multilingual-e5-small",
        bm25_index_path=bm25_path,
        app_env="development",
        elevenlabs_api_key="",
    )

    fake_store = MagicMock()
    fake_store.ping.return_value = True
    fake_store.count.return_value = 1
    with patch("app.dependencies.QdrantStore", return_value=fake_store):
        pipeline = build_pipeline(settings)

    assert isinstance(pipeline.hybrid.dense, CloudDenseRetriever)
    assert pipeline.llm.__class__.__name__ == "ExtractiveAnswerProvider"
    assert pipeline.reranker.__class__.__name__ == "LexicalLightReranker"


def test_cloud_dense_search_delegates_to_inference_store() -> None:
    store = MagicMock()
    store.search_with_inference.return_value = [
        RetrievalResult("A corporation is a legal entity.", 0.8, "d1", "c1")
    ]
    dense = CloudDenseRetriever(store, inference_model="intfloat/multilingual-e5-small", dimension=384)
    out = dense.search("What is a corporation?", top_k=5)
    store.search_with_inference.assert_called_once_with(
        "What is a corporation?",
        model="intfloat/multilingual-e5-small",
        top_k=5,
    )
    assert out.results[0].chunk_id == "c1"
    assert out.embedding_ms >= 0.0
