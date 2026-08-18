from __future__ import annotations

import uuid
import time

from fastapi import APIRouter, HTTPException, Request

from app.core.exceptions import InvalidQueryError, RAGError
from app.models.schemas import QueryRequest, RAGResponse, WarmupResponse

router = APIRouter()


@router.get("/api/rag/warmup", response_model=WarmupResponse)
def rag_warmup(request: Request) -> WarmupResponse:
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline is not initialized")
    started = time.perf_counter()
    qdrant_status = "ok"
    bm25_status = "ok"
    try:
        if hasattr(pipeline.hybrid.bm25, "_ensure_postings"):
            pipeline.hybrid.bm25._ensure_postings()
    except Exception as exc:  # noqa: BLE001
        bm25_status = f"error: {exc}"
    try:
        pipeline.hybrid.dense.search("warmup", top_k=1)
    except Exception as exc:  # noqa: BLE001
        qdrant_status = f"error: {exc}"
    warmup_ms = (time.perf_counter() - started) * 1000
    status_str = "ready" if qdrant_status == "ok" and bm25_status == "ok" else "degraded"
    return WarmupResponse(
        status=status_str,
        warmup_ms=round(warmup_ms, 2),
        qdrant=qdrant_status,
        bm25=bm25_status,
        retrieval_mode=pipeline.settings.retrieval_mode,
    )



@router.post("/api/rag/query", response_model=RAGResponse)
def rag_query(payload: QueryRequest, request: Request) -> RAGResponse:
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline is not initialized")
    request_id = str(uuid.uuid4())
    started_at = getattr(request.state, "request_started_at", time.perf_counter())
    request_parsing_ms = (time.perf_counter() - started_at) * 1000
    try:
        return pipeline.run(
            payload.query,
            request_id=request_id,
            request_parsing_ms=request_parsing_ms,
        )
    except InvalidQueryError as exc:
        raise HTTPException(status_code=400, detail={"error": exc.message, "code": exc.code}) from exc
    except RAGError as exc:
        raise HTTPException(status_code=exc.status_code, detail={"error": exc.message, "code": exc.code}) from exc
