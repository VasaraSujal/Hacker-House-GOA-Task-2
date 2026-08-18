from __future__ import annotations

import logging
import uuid
from typing import Any, Iterable, Sequence

from app.core.exceptions import VectorStoreError
from rag.chunking.base import Chunk
from rag.retrieval.types import RetrievalResult

logger = logging.getLogger(__name__)


def _point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


class QdrantStore:
    """Reusable Qdrant client. Collection dimension matches the embedding model."""

    def __init__(
        self,
        url: str = "http://localhost:6333",
        collection: str = "hh_goa_rag",
        api_key: str | None = None,
        vector_size: int = 384,
        timeout: float = 30.0,
        *,
        cloud_inference: bool = False,
    ) -> None:
        self.url = url
        self.collection = collection
        self.vector_size = vector_size
        self.timeout = timeout
        self.cloud_inference = cloud_inference
        self._client = self._connect(url, api_key, timeout, cloud_inference=cloud_inference)

    @staticmethod
    def _connect(url: str, api_key: str | None, timeout: float, *, cloud_inference: bool = False):
        try:
            import httpx
            from qdrant_client import QdrantClient
        except ImportError as exc:
            raise VectorStoreError("qdrant-client is not installed") from exc
        try:
            kwargs: dict[str, Any] = {
                "url": url,
                "api_key": api_key or None,
                "timeout": timeout,
                "check_compatibility": False,
                "limits": httpx.Limits(
                    max_keepalive_connections=20,
                    max_connections=50,
                    keepalive_expiry=300.0,
                ),
            }
            # cloud_inference is only meaningful against Qdrant Cloud clusters.
            if cloud_inference:
                kwargs["cloud_inference"] = True
            return QdrantClient(**kwargs)
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"Cannot connect to Qdrant at {url}: {exc}") from exc

    def ping(self) -> bool:
        try:
            self._client.get_collections()
            return True
        except Exception:  # noqa: BLE001
            return False

    def ensure_collection(self) -> None:
        from qdrant_client.http import models as qmodels

        try:
            existing = {c.name for c in self._client.get_collections().collections}
            if self.collection in existing:
                info = self._client.get_collection(self.collection)
                current = info.config.params.vectors.size  # type: ignore[union-attr]
                if int(current) != int(self.vector_size):
                    raise VectorStoreError(
                        f"Collection {self.collection} has dim={current}, embedding dim={self.vector_size}. "
                        "Delete the collection or change EMBEDDING_MODEL."
                    )
                return
            logger.info(
                "Creating Qdrant collection",
                extra={"collection": self.collection, "dim": self.vector_size},
            )
            self._client.create_collection(
                collection_name=self.collection,
                vectors_config=qmodels.VectorParams(
                    size=self.vector_size,
                    distance=qmodels.Distance.COSINE,
                ),
            )
        except VectorStoreError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"Failed to ensure Qdrant collection: {exc}") from exc

    def upsert_chunks(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> int:
        if not chunks:
            return 0
        if len(chunks) != len(vectors):
            raise VectorStoreError("chunks and vectors length mismatch")
        from qdrant_client.http import models as qmodels

        points = [
            qmodels.PointStruct(
                id=_point_id(chunk.chunk_id),
                vector=list(map(float, vector)),
                payload=chunk.to_payload(),
            )
            for chunk, vector in zip(chunks, vectors)
        ]
        try:
            self._client.upsert(collection_name=self.collection, points=points, wait=True)
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"Qdrant upsert failed: {exc}") from exc
        return len(points)

    def search(self, vector: Sequence[float], top_k: int = 20) -> list[RetrievalResult]:
        query_vector = list(map(float, vector))
        try:
            # qdrant-client >=1.14 uses query_points; older clients expose search().
            if hasattr(self._client, "query_points"):
                response = self._client.query_points(
                    collection_name=self.collection,
                    query=query_vector,
                    limit=top_k,
                    with_payload=True,
                )
                hits = response.points
            else:
                hits = self._client.search(
                    collection_name=self.collection,
                    query_vector=query_vector,
                    limit=top_k,
                    with_payload=True,
                )
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"Qdrant search failed: {exc}") from exc
        results: list[RetrievalResult] = []
        for hit in hits:
            payload: dict[str, Any] = dict(hit.payload or {})
            text = str(payload.pop("text", ""))
            document_id = str(payload.pop("document_id", ""))
            chunk_id = str(payload.pop("chunk_id", hit.id))
            results.append(
                RetrievalResult(
                    text=text,
                    score=float(hit.score or 0.0),
                    document_id=document_id,
                    chunk_id=str(chunk_id),
                    metadata=payload,
                )
            )
        return results

    def search_with_inference(
        self,
        query_text: str,
        *,
        model: str,
        top_k: int = 20,
    ) -> list[RetrievalResult]:
        """Dense search where Qdrant Cloud embeds the query via hosted inference."""
        if not self.cloud_inference:
            raise VectorStoreError(
                "search_with_inference requires QdrantStore(cloud_inference=True)"
            )
        try:
            from qdrant_client.http import models as qmodels
        except ImportError as exc:
            raise VectorStoreError("qdrant-client is not installed") from exc
        try:
            document = qmodels.Document(text=query_text, model=model)
            if hasattr(self._client, "query_points"):
                response = self._client.query_points(
                    collection_name=self.collection,
                    query=document,
                    limit=top_k,
                    with_payload=True,
                )
                hits = response.points
            else:
                raise VectorStoreError("Installed qdrant-client cannot run Document inference queries")
        except VectorStoreError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"Qdrant inference search failed: {exc}") from exc
        results: list[RetrievalResult] = []
        for hit in hits:
            payload: dict[str, Any] = dict(hit.payload or {})
            text = str(payload.pop("text", ""))
            document_id = str(payload.pop("document_id", ""))
            chunk_id = str(payload.pop("chunk_id", hit.id))
            results.append(
                RetrievalResult(
                    text=text,
                    score=float(hit.score or 0.0),
                    document_id=document_id,
                    chunk_id=str(chunk_id),
                    metadata=payload,
                )
            )
        return results

    def count(self) -> int:
        try:
            return int(self._client.count(self.collection, exact=True).count)
        except Exception:  # noqa: BLE001
            return 0
