from __future__ import annotations

import logging
import threading

from app.core.exceptions import RerankerError
from rag.reranking.base import Reranker
from rag.retrieval.types import RetrievalResult

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()
_CACHE: dict[str, object] = {}


class CrossEncoderReranker(Reranker):
    def __init__(self, model_name: str, device: str = "cpu") -> None:
        if not model_name:
            raise RerankerError("RERANKER_MODEL is empty")
        self.model_name = model_name
        self.device = device
        self._model = self._load()

    def _load(self):
        key = f"{self.model_name}::{self.device}"
        with _LOCK:
            cached = _CACHE.get(key)
            if cached is not None:
                return cached
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:
                raise RerankerError("sentence-transformers is not installed") from exc
            logger.info("Loading reranker", extra={"model": self.model_name})
            try:
                model = CrossEncoder(self.model_name, device=self.device)
            except Exception as exc:  # noqa: BLE001
                raise RerankerError(f"Failed to load reranker {self.model_name}: {exc}") from exc
            _CACHE[key] = model
            return model

    def rerank(self, query: str, candidates: list[RetrievalResult], top_k: int) -> list[RetrievalResult]:
        if not candidates:
            return []
        pairs = [(query, c.text) for c in candidates]
        try:
            scores = self._model.predict(pairs, show_progress_bar=False)
        except Exception as exc:  # noqa: BLE001
            raise RerankerError(f"Reranking failed: {exc}") from exc
        scored = []
        for candidate, score in zip(candidates, scores):
            scored.append(
                RetrievalResult(
                    text=candidate.text,
                    score=float(score),
                    document_id=candidate.document_id,
                    chunk_id=candidate.chunk_id,
                    metadata={**candidate.metadata, "rerank_score": float(score)},
                )
            )
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:top_k]
