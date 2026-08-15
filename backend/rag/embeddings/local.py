from __future__ import annotations

import logging
import threading

import numpy as np

from app.core.exceptions import EmbeddingError
from rag.embeddings.base import EmbeddingProvider

logger = logging.getLogger(__name__)

_MODEL_LOCK = threading.Lock()
_MODEL_CACHE: dict[str, object] = {}


def _cache_key(model_name: str, device: str) -> str:
    return f"{model_name}::{device}"


class LocalEmbeddingProvider(EmbeddingProvider):
    """Sentence-Transformers encoder. Model is loaded once per (name, device)."""

    def __init__(
        self,
        model_name: str,
        *,
        device: str = "cpu",
        normalize: bool = True,
        batch_size: int = 64,
    ) -> None:
        if not model_name:
            raise EmbeddingError("EMBEDDING_MODEL is empty")
        self._model_name = model_name
        self._device = device
        self._normalize = normalize
        self._batch_size = batch_size
        self._is_e5 = "e5" in model_name.lower()
        self._model = self._load()
        self._dimension = int(self._model.get_sentence_embedding_dimension())

    def _load(self):
        key = _cache_key(self._model_name, self._device)
        with _MODEL_LOCK:
            cached = _MODEL_CACHE.get(key)
            if cached is not None:
                return cached
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise EmbeddingError("sentence-transformers is not installed") from exc
            logger.info("Loading embedding model", extra={"model": self._model_name, "device": self._device})
            try:
                model = SentenceTransformer(self._model_name, device=self._device)
            except Exception as exc:  # noqa: BLE001
                raise EmbeddingError(f"Failed to load embedding model {self._model_name}: {exc}") from exc
            _MODEL_CACHE[key] = model
            return model

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model_name

    def _prefix(self, texts: list[str], kind: str) -> list[str]:
        if not self._is_e5:
            return texts
        tag = "query" if kind == "query" else "passage"
        return [f"{tag}: {t}" for t in texts]

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dimension), dtype=np.float32)
        try:
            vectors = self._model.encode(
                self._prefix(texts, "document"),
                batch_size=self._batch_size,
                convert_to_numpy=True,
                normalize_embeddings=self._normalize,
                show_progress_bar=False,
            )
        except Exception as exc:  # noqa: BLE001
            raise EmbeddingError(f"Document embedding failed: {exc}") from exc
        return np.asarray(vectors, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        try:
            vector = self._model.encode(
                self._prefix([text], "query"),
                batch_size=1,
                convert_to_numpy=True,
                normalize_embeddings=self._normalize,
                show_progress_bar=False,
            )[0]
        except Exception as exc:  # noqa: BLE001
            raise EmbeddingError(f"Query embedding failed: {exc}") from exc
        return np.asarray(vector, dtype=np.float32)
