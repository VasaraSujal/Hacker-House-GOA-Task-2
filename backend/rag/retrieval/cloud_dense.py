from __future__ import annotations

import time

from rag.embeddings.cloud_meta import CloudInferenceMetaEmbeddings
from rag.retrieval.dense import DenseSearchOutput
from rag.retrieval.qdrant_store import QdrantStore


class CloudDenseRetriever:
    """Dense retrieval via Qdrant Cloud hosted inference (no local embedding model)."""

    def __init__(
        self,
        store: QdrantStore,
        *,
        inference_model: str,
        dimension: int = 384,
    ) -> None:
        self.store = store
        self.inference_model = inference_model
        self.embeddings = CloudInferenceMetaEmbeddings(inference_model, dimension=dimension)

    def search(self, query: str, top_k: int = 20) -> DenseSearchOutput:
        t0 = time.perf_counter()
        results = self.store.search_with_inference(
            query,
            model=self.inference_model,
            top_k=top_k,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        # Embedding happens inside Qdrant; attribute wall time to embedding for diagnostics
        # and keep search_ms as the remaining residual (0) so hybrid totals stay coherent.
        return DenseSearchOutput(results=results, embedding_ms=elapsed_ms, search_ms=0.0)
