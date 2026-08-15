from __future__ import annotations

import re
from typing import Any

from rag.chunking.base import Chunk
from rag.chunking.metadata import make_chunk, normalize_whitespace, split_sentences

_WORD_RE = re.compile(r"\w+", re.UNICODE)


def _tokens(text: str) -> set[str]:
    return {t.lower() for t in _WORD_RE.findall(text)}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


class SemanticChunker:
    """Group consecutive sentences while they remain lexically related.

    Uses Jaccard overlap as a lightweight baseline so ingestion does not
    require a second embedding model. Can later be swapped for embedding
    cosine similarity without changing the Chunker interface.
    """

    strategy = "semantic"

    def __init__(
        self,
        chunk_size: int = 500,
        overlap: int = 50,
        similarity_threshold: float = 0.35,
    ) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.similarity_threshold = similarity_threshold

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
        if not sentences:
            return []

        groups: list[list[str]] = []
        current: list[str] = []
        current_tokens: set[str] = set()
        current_len = 0

        for sentence in sentences:
            sent_tokens = _tokens(sentence)
            extra = len(sentence) + (1 if current else 0)
            similar = True if not current else _jaccard(current_tokens, sent_tokens) >= self.similarity_threshold
            would_overflow = current and current_len + extra > self.chunk_size
            if current and (would_overflow or not similar):
                groups.append(current)
                current = [sentence]
                current_tokens = set(sent_tokens)
                current_len = len(sentence)
                continue
            current.append(sentence)
            current_tokens |= sent_tokens
            current_len += extra

        if current:
            groups.append(current)

        chunks: list[Chunk] = []
        for i, group in enumerate(groups):
            chunks.append(
                make_chunk(
                    " ".join(group),
                    document_id=document_id,
                    language=language,
                    strategy=self.strategy,
                    position=i,
                    metadata=metadata,
                )
            )
        return chunks
