from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class RetrievalResult:
    text: str
    score: float
    document_id: str
    chunk_id: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_source_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "score": self.score,
            "document_id": self.document_id,
            "chunk_id": self.chunk_id,
            "metadata": self.metadata,
        }
