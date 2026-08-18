from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    # Keep aligned with Settings.max_query_chars / InputGuard.
    query: str = Field(..., min_length=1, max_length=2000)


class SourceDocument(BaseModel):
    text: str
    score: float
    document_id: str
    chunk_id: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class LatencyBreakdown(BaseModel):
    request_parsing_ms: float = 0.0
    query_processing_ms: float = 0.0
    embedding_ms: float = 0.0
    dense_retrieval_ms: float = 0.0
    bm25_ms: float = 0.0
    retrieval_wall_ms: float = 0.0
    fusion_ms: float = 0.0
    relevance_guard_ms: float = 0.0
    reranking_ms: float = 0.0
    context_building_ms: float = 0.0
    generation_ms: float = 0.0
    grounding_ms: float = 0.0
    rag_core_ms: float = 0.0
    component_sum_ms: float = 0.0
    unaccounted_ms: float = 0.0
    total_ms: float = 0.0


class VoiceLatencyBreakdown(LatencyBreakdown):
    audio_validation_ms: float = 0.0
    stt_ms: float = 0.0
    transcript_validation_ms: float = 0.0
    rag_ms: float = 0.0


class RAGResponse(BaseModel):
    query: str
    answer: str
    sources: list[SourceDocument] = Field(default_factory=list)
    grounded: bool = False
    refused: bool = False
    request_id: str | None = None
    latency: LatencyBreakdown = Field(default_factory=LatencyBreakdown)


class VoiceRAGResponse(BaseModel):
    transcript: str
    answer: str
    sources: list[SourceDocument] = Field(default_factory=list)
    grounded: bool = False
    refused: bool = False
    request_id: str | None = None
    latency: VoiceLatencyBreakdown = Field(default_factory=VoiceLatencyBreakdown)


class HealthResponse(BaseModel):
    status: str
    qdrant: str
    embeddings: str
    bm25: str
    elevenlabs_configured: bool
    stt_configured: bool


class WarmupResponse(BaseModel):
    status: str
    warmup_ms: float
    qdrant: str
    bm25: str
    retrieval_mode: str


class ErrorResponse(BaseModel):
    error: str
    code: str
    detail: str | None = None

