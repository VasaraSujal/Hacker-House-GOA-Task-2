from __future__ import annotations

import logging
import time
from typing import Any

import httpx

from app.core.exceptions import STTError
from rag.stt.base import STTProvider, Transcript

logger = logging.getLogger(__name__)
TRANSIENT_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


class ElevenLabsSTTProvider(STTProvider):
    """ElevenLabs batch Speech-to-Text API client for uploaded audio files."""

    def __init__(
        self,
        api_key: str,
        *,
        model: str = "scribe_v2",
        api_base: str = "https://api.elevenlabs.io",
        timeout_s: float = 30.0,
        max_retries: int = 3,
    ) -> None:
        if not api_key:
            raise STTError(
                "ELEVENLABS_API_KEY is not configured",
                status_code=503,
                code="stt_not_configured",
            )
        self.api_key = api_key
        self.model = model
        self.api_base = api_base.rstrip("/")
        self.timeout_s = timeout_s
        self.max_retries = max(1, max_retries)
        self._client = httpx.Client(timeout=timeout_s)

    def close(self) -> None:
        self._client.close()

    def transcribe(
        self,
        audio_bytes: bytes,
        *,
        filename: str,
        content_type: str,
    ) -> Transcript:
        files = {"file": (filename, audio_bytes, content_type)}
        data = {"model_id": self.model}
        started = time.perf_counter()
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self._client.post(
                    f"{self.api_base}/v1/speech-to-text",
                    headers={"xi-api-key": self.api_key},
                    files=files,
                    data=data,
                )
            except httpx.TimeoutException as exc:
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(min(2 ** (attempt - 1), 4))
                    continue
                raise STTError("ElevenLabs STT request timed out", status_code=504, code="stt_timeout") from exc
            except httpx.HTTPError as exc:
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(min(2 ** (attempt - 1), 4))
                    continue
                raise STTError("ElevenLabs STT request failed", status_code=502) from exc

            if response.status_code in TRANSIENT_STATUS and attempt < self.max_retries:
                retry_after = response.headers.get("Retry-After")
                delay = float(retry_after) if retry_after and retry_after.replace(".", "", 1).isdigit() else min(2 ** (attempt - 1), 4)
                time.sleep(delay)
                continue
            if response.status_code >= 400:
                status = 429 if response.status_code == 429 else 502
                raise STTError(
                    f"ElevenLabs STT returned {response.status_code}",
                    status_code=status,
                    code="stt_rate_limited" if status == 429 else "stt_provider_error",
                )
            payload: dict[str, Any] = response.json()
            text = str(payload.get("text") or "").strip()
            if not text:
                raise STTError("ElevenLabs STT returned an empty transcript", status_code=422, code="empty_transcript")
            return Transcript(
                text=text,
                language=payload.get("language_code"),
                latency_ms=(time.perf_counter() - started) * 1000,
                raw={"status": response.status_code},
            )
        raise STTError(f"ElevenLabs STT request failed: {last_error}")
