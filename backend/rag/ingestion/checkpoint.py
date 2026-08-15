from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class IngestCheckpoint:
    dataset_id: str
    config: str
    split: str
    processed_query_ids: list[str] = field(default_factory=list)
    passages_upserted: int = 0
    chunks_upserted: int = 0
    last_query_id: str | None = None

    def processed_set(self) -> set[str]:
        return set(self.processed_query_ids)


def load_checkpoint(path: Path) -> IngestCheckpoint | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return IngestCheckpoint(**data)


def save_checkpoint(path: Path, checkpoint: IngestCheckpoint) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(checkpoint.__dict__, indent=2), encoding="utf-8")
    tmp.replace(path)
