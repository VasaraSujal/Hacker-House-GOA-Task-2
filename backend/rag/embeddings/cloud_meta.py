from __future__ import annotations

import numpy as np

from rag.embeddings.base import EmbeddingProvider


class CloudInferenceMetaEmbeddings(EmbeddingProvider):
    """Metadata-only embedding stub for cloud inference mode.

    Vectors are generated inside Qdrant Cloud via Document inference objects.
    This class exists so health checks and hybrid wiring can report dimension
    without loading Torch or SentenceTransformers.
    """

    def __init__(self, model_name: str, dimension: int = 384) -> None:
        self._model_name = model_name
        self._dimension = int(dimension)

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        raise RuntimeError(
            "Local document embedding is disabled in cloud_dense_sparse mode; "
            "use Qdrant Cloud inference upserts instead."
        )

    def embed_query(self, text: str) -> np.ndarray:
        raise RuntimeError(
            "Local query embedding is disabled in cloud_dense_sparse mode; "
            "dense search uses Qdrant Document inference."
        )
