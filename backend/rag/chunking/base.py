from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(slots=True)
class Chunk:
    text: str
    document_id: str
    chunk_id: str
    language: str
    chunk_strategy: str
    position: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "document_id": self.document_id,
            "chunk_id": self.chunk_id,
            "language": self.language,
            "chunk_strategy": self.chunk_strategy,
            "position": self.position,
            **self.metadata,
        }


class Chunker(Protocol):
    strategy: str

    def chunk(self, text: str, *, document_id: str, language: str, metadata: dict[str, Any] | None = None) -> list[Chunk]:
        ...
