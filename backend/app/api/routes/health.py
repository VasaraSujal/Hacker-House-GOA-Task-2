from __future__ import annotations

import logging

from fastapi import APIRouter, Request

from app.core.config import get_settings
from app.models.schemas import HealthResponse

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    settings = get_settings()
    pipeline = getattr(request.app.state, "pipeline", None)
    qdrant_status = "unknown"
    embeddings_status = "unknown"
    bm25_status = "unknown"
    if pipeline is not None:
        try:
            store = pipeline.hybrid.dense.store
            qdrant_status = "ok" if store.ping() else "unavailable"
        except Exception as exc:  # noqa: BLE001
            qdrant_status = f"error: {exc}"
        try:
            dim = pipeline.hybrid.dense.embeddings.dimension
            embeddings_status = f"ok (dim={dim})"
        except Exception as exc:  # noqa: BLE001
            embeddings_status = f"error: {exc}"
        try:
            n = len(pipeline.hybrid.bm25)
            bm25_status = f"ok ({n} docs)" if n else "empty"
        except Exception as exc:  # noqa: BLE001
            bm25_status = f"error: {exc}"
    else:
        qdrant_status = "pipeline_not_loaded"
        embeddings_status = "pipeline_not_loaded"
        bm25_status = "pipeline_not_loaded"

    overall = "ok" if qdrant_status.startswith("ok") else "degraded"
    return HealthResponse(
        status=overall,
        qdrant=qdrant_status,
        embeddings=embeddings_status,
        bm25=bm25_status,
        elevenlabs_configured=bool(settings.elevenlabs_api_key),
        stt_configured=bool(getattr(request.app.state, "stt", None)),
    )
