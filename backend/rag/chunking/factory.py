from __future__ import annotations

from rag.chunking.base import Chunker
from rag.chunking.fixed import FixedSizeChunker
from rag.chunking.metadata_aware import MetadataAwareChunker
from rag.chunking.semantic import SemanticChunker
from rag.chunking.sentence import SentenceChunker


def get_chunker(
    strategy: str,
    chunk_size: int = 500,
    overlap: int = 50,
    similarity_threshold: float = 0.35,
) -> Chunker:
    name = (strategy or "sentence").lower()
    if name in {"fixed", "fixed-size", "fixed_size"}:
        return FixedSizeChunker(chunk_size=chunk_size, overlap=overlap)
    if name in {"sentence", "sentence-based"}:
        return SentenceChunker(chunk_size=chunk_size, overlap=overlap)
    if name in {"semantic"}:
        return SemanticChunker(
            chunk_size=chunk_size,
            overlap=overlap,
            similarity_threshold=similarity_threshold,
        )
    if name in {"metadata", "metadata-aware"}:
        return MetadataAwareChunker(chunk_size=chunk_size, overlap=overlap)
    raise ValueError(f"Unknown chunk strategy: {strategy}")
