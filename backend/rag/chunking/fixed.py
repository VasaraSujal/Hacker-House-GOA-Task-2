from __future__ import annotations

from typing import Any

from rag.chunking.base import Chunk
from rag.chunking.metadata import char_windows, make_chunk, normalize_whitespace


class FixedSizeChunker:
    strategy = "fixed"

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(
        self,
        text: str,
        *,
        document_id: str,
        language: str,
        metadata: dict[str, Any] | None = None,
    ) -> list[Chunk]:
        cleaned = normalize_whitespace(text)
        windows = char_windows(cleaned, self.chunk_size, self.overlap)
        return [
            make_chunk(
                window,
                document_id=document_id,
                language=language,
                strategy=self.strategy,
                position=i,
                metadata=metadata,
            )
            for i, window in enumerate(windows)
        ]
