from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(slots=True)
class GenerationResult:
    text: str
    latency_ms: float
    model: str
    raw: dict | None = None


class LLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str, *, system_prompt: str | None = None) -> GenerationResult:
        ...
