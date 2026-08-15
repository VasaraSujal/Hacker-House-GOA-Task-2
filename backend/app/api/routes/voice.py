from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Request, UploadFile, status

from app.core.config import get_settings
from app.core.exceptions import InvalidAudioError, InvalidQueryError, RAGError, STTError
from app.core.rate_limit import SlidingWindowRateLimiter
from app.models.schemas import VoiceLatencyBreakdown, VoiceRAGResponse

logger = logging.getLogger(__name__)
router = APIRouter()
_voice_limiter = SlidingWindowRateLimiter(limit=20)

_MIME_BY_EXTENSION = {
    ".wav": {"audio/wav", "audio/x-wav", "application/octet-stream"},
    ".mp3": {"audio/mpeg", "audio/mp3", "application/octet-stream"},
    ".m4a": {"audio/mp4", "audio/x-m4a", "application/octet-stream"},
    ".webm": {"audio/webm", "video/webm", "application/octet-stream"},
}


def _validate_upload(upload: UploadFile, audio: bytes, max_bytes: int) -> None:
    filename = (upload.filename or "").strip()
    if not filename:
        raise InvalidAudioError("Audio filename is required")
    if not audio:
        raise InvalidAudioError("Audio file is empty")
    if len(audio) > max_bytes:
        raise InvalidAudioError(f"Audio file exceeds the configured {max_bytes // (1024 * 1024)} MB limit")
    extension = Path(filename).suffix.lower()
    allowed_types = _MIME_BY_EXTENSION.get(extension)
    if allowed_types is None:
        raise InvalidAudioError("Unsupported audio format. Upload WAV, MP3, M4A, or WebM.")
    # Browsers include codec parameters (for example audio/webm;codecs=opus).
    # Validate the media type while preserving the original value for the STT upload.
    content_type = (upload.content_type or "application/octet-stream").split(";", 1)[0].strip().lower()
    if content_type not in allowed_types:
        raise InvalidAudioError(f"Content type {content_type!r} does not match {extension} audio")


@router.post(
    "/api/voice/query",
    response_model=VoiceRAGResponse,
    summary="Transcribe audio with ElevenLabs and answer via grounded RAG",
    responses={
        400: {"description": "Invalid audio or transcript"},
        422: {"description": "ElevenLabs returned an empty transcript"},
        429: {"description": "ElevenLabs STT rate limited"},
        503: {"description": "Voice RAG is not initialized"},
        504: {"description": "ElevenLabs STT timed out"},
    },
)
def voice_query(
    request: Request,
    audio: UploadFile = File(
        ...,
        description="WAV, MP3, M4A, or WebM audio supported by ElevenLabs STT.",
    ),
) -> VoiceRAGResponse:
    settings = get_settings()
    limiter = getattr(request.app.state, "voice_limiter", _voice_limiter)
    client_host = request.client.host if request.client else "unknown"
    if not limiter.allow(client_host):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "error": "The service is temporarily rate-limited. Please try again shortly.",
                "code": "voice_rate_limited",
            },
        )
    request_id = str(uuid.uuid4())
    started = time.perf_counter()
    validation_started = time.perf_counter()
    max_bytes = settings.max_audio_size_mb * 1024 * 1024
    # Read only one byte over the cap, avoiding unbounded uploads in RAM.
    audio_bytes = audio.file.read(max_bytes + 1)
    try:
        _validate_upload(audio, audio_bytes, max_bytes)
    except InvalidAudioError as exc:
        raise HTTPException(status_code=exc.status_code, detail={"error": exc.message, "code": exc.code}) from exc
    audio_validation_ms = (time.perf_counter() - validation_started) * 1000

    pipeline = getattr(request.app.state, "pipeline", None)
    stt = getattr(request.app.state, "stt", None)
    if pipeline is None or stt is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Voice RAG is not initialized")

    try:
        transcript = stt.transcribe(
            audio_bytes,
            filename=audio.filename or "audio",
            content_type=audio.content_type or "application/octet-stream",
        )
        transcript_validation_started = time.perf_counter()
        # Reuse exactly the existing input validation policy before retrieval.
        pipeline.input_guard.validate(transcript.text)
        transcript_validation_ms = (time.perf_counter() - transcript_validation_started) * 1000
        rag_started = time.perf_counter()
        rag = pipeline.run(transcript.text, request_id=request_id)
        rag_ms = (time.perf_counter() - rag_started) * 1000
    except (InvalidQueryError, STTError) as exc:
        logger.info(
            "Voice request rejected",
            extra={
                "request_id": request_id,
                "audio_filename": audio.filename,
                "audio_bytes": len(audio_bytes),
                "content_type": audio.content_type,
                "error_code": exc.code,
            },
        )
        raise HTTPException(status_code=exc.status_code, detail={"error": exc.message, "code": exc.code}) from exc
    except RAGError as exc:
        raise HTTPException(status_code=exc.status_code, detail={"error": exc.message, "code": exc.code}) from exc

    rag_latency = rag.latency.model_dump()
    rag_latency.pop("total_ms", None)
    rag_latency.pop("component_sum_ms", None)
    rag_latency.pop("unaccounted_ms", None)
    total_ms = (time.perf_counter() - started) * 1000
    voice_component_sum_ms = (
        audio_validation_ms
        + transcript.latency_ms
        + transcript_validation_ms
        + rag_ms
    )
    latency = VoiceLatencyBreakdown(
        **rag_latency,
        audio_validation_ms=round(audio_validation_ms, 3),
        stt_ms=round(transcript.latency_ms, 3),
        transcript_validation_ms=round(transcript_validation_ms, 3),
        rag_ms=round(rag_ms, 3),
        component_sum_ms=round(voice_component_sum_ms, 3),
        unaccounted_ms=round(max(0.0, total_ms - voice_component_sum_ms), 3),
        total_ms=round(total_ms, 3),
    )
    logger.info(
        "Voice RAG completed",
        extra={
            "request_id": request_id,
            "audio_filename": audio.filename,
            "audio_bytes": len(audio_bytes),
            "content_type": audio.content_type,
            "stt_ms": latency.stt_ms,
            "transcript_length": len(transcript.text),
            "rag_ms": latency.rag_ms,
            "generation_ms": latency.generation_ms,
            "total_ms": latency.total_ms,
            "grounded": rag.grounded,
            "refused": rag.refused,
        },
    )
    return VoiceRAGResponse(
        transcript=transcript.text,
        answer=rag.answer,
        sources=rag.sources,
        grounded=rag.grounded,
        refused=rag.refused,
        request_id=request_id,
        latency=latency,
    )
