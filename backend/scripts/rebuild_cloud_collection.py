"""Rebuild a Qdrant Cloud production collection using hosted inference.

Reads payloads from a source collection (local or cloud) WITHOUT deleting it,
creates a separate target collection, and upserts Document inference objects so
query-time and document-time embeddings use the same hosted model.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels


def _client(url: str, api_key: str, *, cloud_inference: bool) -> QdrantClient:
    kwargs = {
        "url": url,
        "api_key": api_key or None,
        "timeout": float(os.getenv("QDRANT_REBUILD_TIMEOUT_S", "120")),
        "check_compatibility": False,
    }
    if cloud_inference:
        kwargs["cloud_inference"] = True
    return QdrantClient(**kwargs)


def main() -> int:
    source_url = os.getenv("SOURCE_QDRANT_URL", "http://127.0.0.1:6333").strip()
    source_key = os.getenv("SOURCE_QDRANT_API_KEY", "").strip()
    source_collection = os.getenv("SOURCE_QDRANT_COLLECTION", "hh_goa_rag").strip()

    target_url = os.getenv("TARGET_QDRANT_URL", "").strip()
    target_key = os.getenv("TARGET_QDRANT_API_KEY", "").strip()
    target_collection = os.getenv("TARGET_QDRANT_COLLECTION", "hh_goa_voice_rag_prod").strip()
    inference_model = os.getenv(
        "QDRANT_INFERENCE_MODEL",
        "intfloat/multilingual-e5-small",
    ).strip()
    dimension = int(os.getenv("QDRANT_INFERENCE_DIMENSION", "384"))
    batch_size = int(os.getenv("QDRANT_REBUILD_BATCH_SIZE", "32"))

    if not target_url:
        print("TARGET_QDRANT_URL is required", file=sys.stderr)
        return 2
    if not target_key:
        print("TARGET_QDRANT_API_KEY is required", file=sys.stderr)
        return 2
    if source_url.rstrip("/") == target_url.rstrip("/") and source_collection == target_collection:
        print("Source and target are identical; refusing rebuild", file=sys.stderr)
        return 2

    source = _client(source_url, source_key, cloud_inference=False)
    target = _client(target_url, target_key, cloud_inference=True)

    source_count = int(source.count(source_collection, exact=True).count)
    print(
        f"Source {source_collection} points={source_count}; "
        f"target={target_collection}; model={inference_model}"
    )

    existing = {item.name for item in target.get_collections().collections}
    if target_collection in existing:
        existing_count = int(target.count(target_collection, exact=True).count)
        if existing_count > 0:
            print(
                f"Target collection {target_collection} already has {existing_count} points. "
                "Refusing to overwrite. Delete it manually in Qdrant Cloud if a rebuild is intended.",
                file=sys.stderr,
            )
            return 2
        info = target.get_collection(target_collection)
        vectors = info.config.params.vectors
        if int(vectors.size) != dimension or vectors.distance != qmodels.Distance.COSINE:
            print("Empty target collection has incompatible vector config", file=sys.stderr)
            return 2
    else:
        target.create_collection(
            collection_name=target_collection,
            vectors_config=qmodels.VectorParams(
                size=dimension,
                distance=qmodels.Distance.COSINE,
            ),
        )
        print(f"Created collection {target_collection} dim={dimension} Cosine")

    copied = 0
    offset = None
    while True:
        points, offset = source.scroll(
            collection_name=source_collection,
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        if not points:
            break
        upsert_points = []
        for point in points:
            payload = dict(point.payload or {})
            text = str(payload.get("text") or "").strip()
            if not text:
                continue
            upsert_points.append(
                qmodels.PointStruct(
                    id=point.id,
                    vector=qmodels.Document(text=text, model=inference_model),
                    payload=payload,
                )
            )
        if upsert_points:
            target.upsert(collection_name=target_collection, points=upsert_points, wait=True)
            copied += len(upsert_points)
            print(f"Upserted {copied}/{source_count} points via cloud inference")
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
    required = {"text", "document_id", "chunk_id"}
    if not sample or not required.issubset(set(sample[0].payload or {})):
        print("Verification failed: target payload schema sample is incomplete", file=sys.stderr)
        return 1

    # Smoke query using the same hosted model
    probe = target.query_points(
        collection_name=target_collection,
        query=qmodels.Document(text="What is a corporation?", model=inference_model),
        limit=3,
        with_payload=True,
    )
    print(
        f"Rebuild verified: collection={target_collection} points={target_count} "
        f"model={inference_model} probe_hits={len(probe.points)}"
    )
    for hit in probe.points[:3]:
        payload = hit.payload or {}
        preview = str(payload.get("text", ""))[:120].replace("\n", " ")
        print(f"  score={hit.score:.4f} chunk={payload.get('chunk_id')} text={preview}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
