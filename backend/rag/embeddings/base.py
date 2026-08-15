from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class EmbeddingProvider(ABC):
    """Local or remote embedding backend. Loaded once and reused."""

    @property
    @abstractmethod
    def dimension(self) -> int: ...

    @property
    @abstractmethod
    def model_name(self) -> str: ...

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Batch-embed passages/chunks."""

    @abstractmethod
    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single search query."""
