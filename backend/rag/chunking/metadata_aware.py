from __future__ import annotations

from rag.chunking.sentence import SentenceChunker


class MetadataAwareChunker(SentenceChunker):
    """Sentence packing with an explicit strategy label.

    All chunkers preserve metadata; this class exists so experiments can
    distinguish metadata-aware indexing from the sentence baseline.
    """

    strategy = "metadata"
