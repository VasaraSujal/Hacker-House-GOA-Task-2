from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
WHITESPACE_RE = re.compile(r"\s+")


@dataclass(slots=True)
class Passage:
    document_id: str
    query_id: str
    passage_index: int
    text: str
    language: str
    is_selected: bool
    source_query: str
    metadata: dict[str, Any] = field(default_factory=dict)


def clean_text(text: str | None) -> str:
    if text is None:
        return ""
    cleaned = CONTROL_RE.sub(" ", str(text))
    cleaned = cleaned.replace("\u00a0", " ")
    cleaned = WHITESPACE_RE.sub(" ", cleaned).strip()
    return cleaned


def is_usable_text(text: str, min_chars: int = 20) -> bool:
    return bool(text) and len(text) >= min_chars


def preprocess_query(query: str) -> str:
    return clean_text(query)
