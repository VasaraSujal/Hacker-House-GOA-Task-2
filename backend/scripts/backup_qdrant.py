"""Create and download a local Qdrant collection snapshot."""

from __future__ import annotations

import os
from pathlib import Path

import httpx


def main() -> int:
    base_url = os.getenv("SOURCE_QDRANT_URL", "http://127.0.0.1:6333").rstrip("/")
    collection = os.getenv("SOURCE_QDRANT_COLLECTION", "hh_goa_rag")
    api_key = os.getenv("SOURCE_QDRANT_API_KEY", "")
    output_dir = Path(os.getenv("QDRANT_BACKUP_DIR", "../backups/qdrant")).resolve()
    headers = {"api-key": api_key} if api_key else {}

    with httpx.Client(base_url=base_url, headers=headers, timeout=120.0) as client:
        response = client.post(f"/collections/{collection}/snapshots")
        response.raise_for_status()
        result = response.json()["result"]
        snapshot_name = result["name"]
        download = client.get(f"/collections/{collection}/snapshots/{snapshot_name}")
        download.raise_for_status()

    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / snapshot_name
    target.write_bytes(download.content)
    print(f"Saved Qdrant snapshot: {target} ({target.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
