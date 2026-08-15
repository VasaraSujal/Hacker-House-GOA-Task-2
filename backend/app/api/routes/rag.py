from __future__ import annotations

import uuid
import time

from fastapi import APIRouter, HTTPException, Request

from app.core.exceptions import InvalidQueryError, RAGError
from app.models.schemas import QueryRequest, RAGResponse

router = APIRouter()


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
