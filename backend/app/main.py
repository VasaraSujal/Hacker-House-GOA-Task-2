from __future__ import annotations

import logging
import time

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes.health import router as health_router
from app.api.routes.rag import router as rag_router
from app.api.routes.voice import router as voice_router
from app.core.config import get_settings
from app.core.exceptions import RAGError
from app.core.logging import configure_logging
from app.core.rate_limit import SlidingWindowRateLimiter
from app.dependencies import build_pipeline, try_build_pipeline
from rag.stt.elevenlabs import ElevenLabsSTTProvider

logger = logging.getLogger(__name__)


def create_app(*, load_pipeline: bool = True) -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)
    app = FastAPI(
        title="HH Goa Voice RAG",
        description="ElevenLabs Speech-to-Text plus grounded hybrid RAG backend.",
        version="0.6.0",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type"],
    )
    app.state.voice_limiter = SlidingWindowRateLimiter(
        limit=settings.voice_rate_limit_per_minute,
        window_seconds=60.0,
    )

    @app.middleware("http")
    async def _request_timer(request: Request, call_next):
        request.state.request_started_at = time.perf_counter()
        return await call_next(request)

    @app.on_event("startup")
    def _startup() -> None:
        if not load_pipeline:
            return
        production = settings.app_env.lower() == "production"
        pipeline = build_pipeline(settings) if production else try_build_pipeline(settings)
        app.state.pipeline = pipeline
        try:
            app.state.stt = ElevenLabsSTTProvider(
                settings.elevenlabs_api_key,
                model=settings.elevenlabs_stt_model,
                api_base=settings.elevenlabs_api_base,
                timeout_s=settings.elevenlabs_stt_timeout_s,
                max_retries=settings.elevenlabs_stt_max_retries,
            )
        except RAGError as exc:
            app.state.stt = None
            if production:
                raise RuntimeError("STT configuration is required in production") from exc
            logger.warning("STT not configured at startup; /api/voice/query will be unavailable")
        if pipeline is None:
            if production:
                raise RuntimeError("RAG pipeline initialization failed in production")
            logger.warning("RAG pipeline not loaded at startup; /health will report degraded")

    @app.exception_handler(RAGError)
    async def _rag_error(_: Request, exc: RAGError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": exc.message, "code": exc.code},
        )

    app.include_router(health_router)
    app.include_router(rag_router)
    app.include_router(voice_router)
    return app


app = create_app()
