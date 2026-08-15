from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(slots=True)
class Transcript:
    text: str
    language: str | None = None
    latency_ms: float = 0.0
    raw: dict | None = None


class STTProvider(ABC):
    """Provider boundary: audio bytes in, validated transcript out."""

    @abstractmethod
    def transcribe(
        self,
        audio_bytes: bytes,
        *,
        filename: str,
        content_type: str,
    ) -> Transcript:
        ...
