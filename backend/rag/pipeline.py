from __future__ import annotations

import logging
import time
import uuid

from app.core.config import Settings
from app.core.exceptions import GenerationError
from app.models.schemas import LatencyBreakdown, RAGResponse, SourceDocument
from rag.context.builder import ContextBuilder
from rag.generation.base import LLMProvider
from rag.generation.prompts import REFUSAL_MESSAGE, SYSTEM_PROMPT, build_user_prompt
from rag.guardrails.grounding_guard import GroundingGuard
from rag.guardrails.input_guard import InputGuard
from rag.guardrails.relevance_guard import RelevanceGuard
from rag.guardrails.coverage_guard import LexicalCoverageGuard
from rag.ingestion.cleaner import preprocess_query
from rag.reranking.base import Reranker
from rag.retrieval.hybrid import HybridRetriever
from rag.retrieval.types import RetrievalResult

logger = logging.getLogger(__name__)


class RAGPipeline:
    """Orchestrates Stage 1 text RAG. Stage 2 will pass STT transcripts into run()."""

    def __init__(
        self,
        *,
        settings: Settings,
        hybrid: HybridRetriever,
        reranker: Reranker,
        llm: LLMProvider,
        input_guard: InputGuard | None = None,
        relevance_guard: RelevanceGuard | None = None,
        grounding_guard: GroundingGuard | None = None,
        context_builder: ContextBuilder | None = None,
        coverage_guard: LexicalCoverageGuard | None = None,
    ) -> None:
        self.settings = settings
        self.hybrid = hybrid
        self.reranker = reranker
        self.llm = llm
        self.input_guard = input_guard or InputGuard(max_chars=settings.max_query_chars)
        self.relevance_guard = relevance_guard or RelevanceGuard(
            min_results=settings.relevance_min_results,
            min_score=settings.relevance_min_score,
        )
        self.grounding_guard = grounding_guard or GroundingGuard(
            min_overlap=settings.grounding_min_overlap,
            refusal_message=settings.refusal_message,
        )
        self.context_builder = context_builder or ContextBuilder(max_chars=settings.max_context_chars)
        self.coverage_guard = coverage_guard or LexicalCoverageGuard()

    def run(
        self,
        query: str,
        request_id: str | None = None,
        *,
        request_parsing_ms: float = 0.0,
    ) -> RAGResponse:
        request_id = request_id or str(uuid.uuid4())
        t_all = time.perf_counter()
        query_started = time.perf_counter()
        cleaned = self.input_guard.validate(query)
        cleaned = preprocess_query(cleaned)
        query_processing_ms = (time.perf_counter() - query_started) * 1000

        hybrid_out = self.hybrid.search(
            cleaned,
            dense_top_k=self.settings.dense_top_k,
            bm25_top_k=self.settings.bm25_top_k,
            hybrid_top_k=self.settings.hybrid_top_k,
        )
        # RRF scores are small (~1/(k+rank)); use dense cosine scores for relevance when available.
        relevance_pool = hybrid_out.dense or hybrid_out.bm25 or hybrid_out.results
        relevance_started = time.perf_counter()
        relevance = self.relevance_guard.check(relevance_pool)
        relevance_guard_ms = (time.perf_counter() - relevance_started) * 1000
        if not relevance.ok:
            pipeline_ms = (time.perf_counter() - t_all) * 1000
            total_ms = request_parsing_ms + pipeline_ms
            logger.info(
                "RAG refused at relevance guard",
                extra={
                    "request_id": request_id,
                    "query_length": len(cleaned),
                    "n_candidates": relevance.n_results,
                    "reason": relevance.reason,
                    "max_score": relevance.max_score,
                    "total_ms": round(total_ms, 2),
                },
            )
            return self._refusal(
                cleaned,
                request_id,
                _build_latency(
                    request_parsing_ms=request_parsing_ms,
                    query_processing_ms=query_processing_ms,
                    embedding_ms=hybrid_out.embedding_ms,
                    dense_retrieval_ms=hybrid_out.dense_retrieval_ms,
                    bm25_ms=hybrid_out.bm25_ms,
                    retrieval_wall_ms=hybrid_out.retrieval_wall_ms,
                    fusion_ms=hybrid_out.fusion_ms,
                    relevance_guard_ms=relevance_guard_ms,
                    total_ms=total_ms,
                ),
            )

        if (self.settings.answer_mode or "").lower() == "extractive":
            coverage = self.coverage_guard.check(cleaned, hybrid_out.results, dense_results=hybrid_out.dense)
            if not coverage.ok:
                pipeline_ms = (time.perf_counter() - t_all) * 1000
                total_ms = request_parsing_ms + pipeline_ms
                logger.info(
                    "RAG refused at lexical coverage guard",
                    extra={
                        "request_id": request_id,
                        "reason": coverage.reason,
                        "best_overlap": coverage.best_overlap,
                        "total_ms": round(total_ms, 2),
                    },
                )
                return self._refusal(
                    cleaned,
                    request_id,
                    _build_latency(
                        request_parsing_ms=request_parsing_ms,
                        query_processing_ms=query_processing_ms,
                        embedding_ms=hybrid_out.embedding_ms,
                        dense_retrieval_ms=hybrid_out.dense_retrieval_ms,
                        bm25_ms=hybrid_out.bm25_ms,
                        retrieval_wall_ms=hybrid_out.retrieval_wall_ms,
                        fusion_ms=hybrid_out.fusion_ms,
                        relevance_guard_ms=relevance_guard_ms,
                        total_ms=total_ms,
                    ),
                    sources=hybrid_out.results[: self.settings.rerank_top_k],
                )

        t_rerank = time.perf_counter()
        reranked = self.reranker.rerank(cleaned, hybrid_out.results, self.settings.rerank_top_k)
        rerank_ms = (time.perf_counter() - t_rerank) * 1000
        if not reranked:
            reranked = hybrid_out.results[: self.settings.rerank_top_k]

        context_started = time.perf_counter()
        context, selected = self.context_builder.build(reranked)
        context_building_ms = (time.perf_counter() - context_started) * 1000
        user_prompt = build_user_prompt(cleaned, context)

        try:
            generation_started = time.perf_counter()
            generation = self.llm.generate(user_prompt, system_prompt=SYSTEM_PROMPT)
            answer = generation.text.strip()
            generation_ms = (time.perf_counter() - generation_started) * 1000
        except GenerationError as exc:
            logger.error("Generation failed", extra={"request_id": request_id, "error": str(exc)})
            pipeline_ms = (time.perf_counter() - t_all) * 1000
            total_ms = request_parsing_ms + pipeline_ms
            return self._refusal(
                cleaned,
                request_id,
                _build_latency(
                    request_parsing_ms=request_parsing_ms,
                    query_processing_ms=query_processing_ms,
                    embedding_ms=hybrid_out.embedding_ms,
                    dense_retrieval_ms=hybrid_out.dense_retrieval_ms,
                    bm25_ms=hybrid_out.bm25_ms,
                    retrieval_wall_ms=hybrid_out.retrieval_wall_ms,
                    fusion_ms=hybrid_out.fusion_ms,
                    relevance_guard_ms=relevance_guard_ms,
                    reranking_ms=rerank_ms,
                    context_building_ms=context_building_ms,
                    total_ms=total_ms,
                ),
                sources=selected,
            )

        grounding_started = time.perf_counter()
        grounding = self.grounding_guard.check(answer, context)
        grounding_ms = (time.perf_counter() - grounding_started) * 1000
        if not grounding.grounded:
            answer = self.settings.refusal_message
            grounded = False
            refused = True
        else:
            grounded = True
            refused = answer.strip() == self.settings.refusal_message

        pipeline_ms = (time.perf_counter() - t_all) * 1000
        total_ms = request_parsing_ms + pipeline_ms
        latency = _build_latency(
            request_parsing_ms=request_parsing_ms,
            query_processing_ms=query_processing_ms,
            embedding_ms=hybrid_out.embedding_ms,
            dense_retrieval_ms=hybrid_out.dense_retrieval_ms,
            bm25_ms=hybrid_out.bm25_ms,
            retrieval_wall_ms=hybrid_out.retrieval_wall_ms,
            fusion_ms=hybrid_out.fusion_ms,
            relevance_guard_ms=relevance_guard_ms,
            reranking_ms=rerank_ms,
            context_building_ms=context_building_ms,
            generation_ms=generation_ms,
            grounding_ms=grounding_ms,
            total_ms=total_ms,
        )
        logger.info(
            "RAG completed",
            extra={
                "request_id": request_id,
                "query_length": len(cleaned),
                "n_candidates": len(hybrid_out.results),
                "n_context_chunks": len(selected),
                "embedding_ms": latency.embedding_ms,
                "dense_retrieval_ms": latency.dense_retrieval_ms,
                "bm25_ms": latency.bm25_ms,
                "fusion_ms": latency.fusion_ms,
                "retrieval_wall_ms": latency.retrieval_wall_ms,
                "reranking_ms": latency.reranking_ms,
                "context_building_ms": latency.context_building_ms,
                "generation_ms": latency.generation_ms,
                "grounding_ms": latency.grounding_ms,
                "total_ms": latency.total_ms,
                "grounded": grounded,
            },
        )
        return RAGResponse(
            query=cleaned,
            answer=answer,
            sources=[_to_source(s) for s in selected],
            grounded=grounded,
            refused=refused,
            request_id=request_id,
            latency=latency,
        )

    def _refusal(
        self,
        query: str,
        request_id: str,
        latency: LatencyBreakdown,
        sources: list[RetrievalResult] | None = None,
    ) -> RAGResponse:
        return RAGResponse(
            query=query,
            answer=self.settings.refusal_message,
            sources=[_to_source(s) for s in (sources or [])],
            grounded=True,
            refused=True,
            request_id=request_id,
            latency=latency,
        )


def _to_source(result: RetrievalResult) -> SourceDocument:
    return SourceDocument(
        text=result.text,
        score=result.score,
        document_id=result.document_id,
        chunk_id=result.chunk_id,
        metadata=result.metadata,
    )


def _build_latency(
    *,
    request_parsing_ms: float = 0.0,
    query_processing_ms: float = 0.0,
    embedding_ms: float = 0.0,
    dense_retrieval_ms: float = 0.0,
    bm25_ms: float = 0.0,
    retrieval_wall_ms: float = 0.0,
    fusion_ms: float = 0.0,
    relevance_guard_ms: float = 0.0,
    reranking_ms: float = 0.0,
    context_building_ms: float = 0.0,
    generation_ms: float = 0.0,
    grounding_ms: float = 0.0,
    total_ms: float = 0.0,
) -> LatencyBreakdown:
    # embedding/dense/BM25 are diagnostic subcomponents of retrieval_wall_ms and
    # are intentionally not added again, otherwise parallel retrieval double-counts.
    rag_core_ms = (
        query_processing_ms
        + retrieval_wall_ms
        + fusion_ms
        + relevance_guard_ms
        + reranking_ms
        + context_building_ms
        + grounding_ms
    )
    component_sum_ms = request_parsing_ms + rag_core_ms + generation_ms
    unaccounted_ms = max(0.0, total_ms - component_sum_ms)
    values = {
        "request_parsing_ms": request_parsing_ms,
        "query_processing_ms": query_processing_ms,
        "embedding_ms": embedding_ms,
        "dense_retrieval_ms": dense_retrieval_ms,
        "bm25_ms": bm25_ms,
        "retrieval_wall_ms": retrieval_wall_ms,
        "fusion_ms": fusion_ms,
        "relevance_guard_ms": relevance_guard_ms,
        "reranking_ms": reranking_ms,
        "context_building_ms": context_building_ms,
        "generation_ms": generation_ms,
        "grounding_ms": grounding_ms,
        "rag_core_ms": rag_core_ms,
        "component_sum_ms": component_sum_ms,
        "unaccounted_ms": unaccounted_ms,
        "total_ms": total_ms,
    }
    return LatencyBreakdown(**{key: round(value, 3) for key, value in values.items()})
