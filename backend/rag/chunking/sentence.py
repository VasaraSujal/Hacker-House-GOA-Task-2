from __future__ import annotations

from typing import Any

from rag.chunking.base import Chunk
from rag.chunking.metadata import make_chunk, normalize_whitespace, pack_sentences, split_sentences


class SentenceChunker:
    strategy = "sentence"

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap
        self._overlap_sentences = 1 if overlap > 0 else 0

    def chunk(
        self,
        text: str,
        *,
        document_id: str,
        language: str,
        metadata: dict[str, Any] | None = None,
    ) -> list[Chunk]:
        cleaned = normalize_whitespace(text)
        sentences = split_sentences(cleaned)
        packed = pack_sentences(sentences, self.chunk_size, self._overlap_sentences)
        return [
            make_chunk(
                piece,
                document_id=document_id,
                language=language,
                strategy=self.strategy,
                position=i,
                metadata=metadata,
            )
            for i, piece in enumerate(packed)
        ]
