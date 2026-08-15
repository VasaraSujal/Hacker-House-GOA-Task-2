"""Copy and verify a Qdrant collection without deleting the source."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels


def _client(url: str, api_key: str) -> QdrantClient:
    return QdrantClient(
        url=url,
        api_key=api_key or None,
        timeout=float(os.getenv("QDRANT_MIGRATION_TIMEOUT_S", "60")),
        check_compatibility=False,
    )


def main() -> int:
    source_url = os.getenv("SOURCE_QDRANT_URL", "http://127.0.0.1:6333")
    source_key = os.getenv("SOURCE_QDRANT_API_KEY", "")
    target_url = os.getenv("TARGET_QDRANT_URL", "").strip()
    target_key = os.getenv("TARGET_QDRANT_API_KEY", "")
    source_collection = os.getenv("SOURCE_QDRANT_COLLECTION", "hh_goa_rag")
    target_collection = os.getenv("TARGET_QDRANT_COLLECTION", source_collection)
    batch_size = int(os.getenv("QDRANT_MIGRATION_BATCH_SIZE", "256"))

    if not target_url:
        print("TARGET_QDRANT_URL is required", file=sys.stderr)
        return 2
    if source_url.rstrip("/") == target_url.rstrip("/") and source_collection == target_collection:
        print("Source and target are identical; refusing migration", file=sys.stderr)
        return 2

    source = _client(source_url, source_key)
    target = _client(target_url, target_key)
    source_info = source.get_collection(source_collection)
    source_count = int(source.count(source_collection, exact=True).count)
    vectors = source_info.config.params.vectors
    if not hasattr(vectors, "size"):
        print("Named/multi-vector collections are not supported by this migration script", file=sys.stderr)
        return 2

    existing = {item.name for item in target.get_collections().collections}
    if target_collection in existing:
        target_info = target.get_collection(target_collection)
        target_vectors = target_info.config.params.vectors
        if int(target_vectors.size) != int(vectors.size) or target_vectors.distance != vectors.distance:
            print("Target collection vector configuration does not match source", file=sys.stderr)
            return 2
    else:
        target.create_collection(
            collection_name=target_collection,
            vectors_config=qmodels.VectorParams(size=int(vectors.size), distance=vectors.distance),
        )

    copied = 0
    offset = None
    while True:
        points, offset = source.scroll(
            collection_name=source_collection,
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )
        if not points:
            break
        target.upsert(
            collection_name=target_collection,
            points=[
                qmodels.PointStruct(id=point.id, vector=point.vector, payload=point.payload)
                for point in points
            ],
            wait=True,
        )
        copied += len(points)
        print(f"Copied {copied}/{source_count} points")
        if offset is None:
            break

    target_count = int(target.count(target_collection, exact=True).count)
    if target_count != source_count:
        print(
            f"Verification failed: source={source_count}, target={target_count}",
            file=sys.stderr,
        )
        return 1

    sample, _ = target.scroll(
        collection_name=target_collection,
        limit=1,
        with_payload=True,
        with_vectors=False,
    )
    required_payload = {"text", "document_id", "chunk_id"}
    if not sample or not required_payload.issubset(set(sample[0].payload or {})):
        print("Verification failed: target payload schema sample is incomplete", file=sys.stderr)
        return 1

    print(
        f"Migration verified: collection={target_collection} points={target_count} "
        f"dimension={vectors.size} distance={vectors.distance}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
