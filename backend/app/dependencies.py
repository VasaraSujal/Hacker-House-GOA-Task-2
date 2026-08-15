from __future__ import annotations

import logging

from app.core.config import Settings
from rag.embeddings.local import LocalEmbeddingProvider
from rag.generation.elevenlabs import ElevenLabsProvider
from rag.pipeline import RAGPipeline
from rag.reranking.cross_encoder import CrossEncoderReranker
from rag.reranking.identity import IdentityReranker
from rag.retrieval.bm25 import BM25Index
from rag.retrieval.dense import DenseRetriever
from rag.retrieval.hybrid import HybridRetriever
from rag.retrieval.qdrant_store import QdrantStore

logger = logging.getLogger(__name__)

_PIPELINE: RAGPipeline | None = None


def build_pipeline(settings: Settings) -> RAGPipeline:
    embeddings = LocalEmbeddingProvider(
        settings.embedding_model,
        device=settings.embedding_device,
        normalize=settings.embedding_normalize,
        batch_size=settings.embedding_batch_size,
    )
    store = QdrantStore(
        url=settings.qdrant_url,
        collection=settings.qdrant_collection,
        api_key=settings.qdrant_api_key,
        vector_size=embeddings.dimension,
        timeout=settings.qdrant_timeout_s,
    )
    bm25 = BM25Index.load(settings.bm25_index_path)
    if settings.app_env.lower() == "production":
        if not store.ping():
            raise RuntimeError("Qdrant is unavailable during production startup")
        if store.count() <= 0:
            raise RuntimeError("Qdrant production collection is empty")
        if len(bm25) <= 0:
            raise RuntimeError("Persisted BM25 index is missing or empty")
    dense = DenseRetriever(embeddings, store)
    hybrid = HybridRetriever(
        dense,
        bm25,
        fusion_method=settings.fusion_method,
        rrf_k=settings.rrf_k,
        dense_weight=settings.dense_weight,
        bm25_weight=settings.bm25_weight,
        parallel=settings.parallel_retrieval,
    )
    reranker = (
        CrossEncoderReranker(settings.reranker_model, device=settings.reranker_device)
        if settings.enable_reranker
        else IdentityReranker()
    )
    llm = ElevenLabsProvider(
        settings.elevenlabs_api_key,
        model=settings.elevenlabs_model,
        api_base=settings.elevenlabs_api_base,
        agent_id=settings.elevenlabs_agent_id,
        timeout_s=settings.elevenlabs_timeout_s,
        max_retries=settings.elevenlabs_max_retries,
    )
    return RAGPipeline(settings=settings, hybrid=hybrid, reranker=reranker, llm=llm)


def try_build_pipeline(settings: Settings) -> RAGPipeline | None:
    global _PIPELINE
    if _PIPELINE is not None:
        return _PIPELINE
    try:
        _PIPELINE = build_pipeline(settings)
        return _PIPELINE
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not initialize RAG pipeline: %s", exc)
        return None
