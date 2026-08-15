from __future__ import annotations

import hashlib
import re
from typing import Any, Iterable

from rag.chunking.base import Chunk

WHITESPACE_RE = re.compile(r"\s+")
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def stable_id(*parts: str) -> str:
    joined = "|".join(parts)
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()[:16]


def normalize_whitespace(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text).strip()


def clean_text(text: str | None) -> str:
    if not text:
        return ""
    cleaned = CONTROL_RE.sub(" ", str(text))
    cleaned = cleaned.replace("\u00a0", " ")
    return normalize_whitespace(cleaned)


def char_windows(text: str, size: int, overlap: int) -> list[str]:
    if size <= 0:
        raise ValueError("chunk size must be positive")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap must be >= 0 and < chunk size")
    if not text:
        return []
    if len(text) <= size:
        return [text]
    step = size - overlap
    windows: list[str] = []
    start = 0
    while start < len(text):
        windows.append(text[start : start + size])
        start += step
    return windows


def split_sentences(text: str) -> list[str]:
    """Lightweight multilingual sentence splitter (Latin + Indic danda)."""
    if not text:
        return []
    parts = re.split(r"(?<=[\.!\?।॥؟])\s+", text)
    sentences = [normalize_whitespace(p) for p in parts if p and p.strip()]
    return sentences or [text]


def pack_sentences(sentences: Iterable[str], max_chars: int, overlap_sentences: int = 1) -> list[str]:
    items = [s for s in sentences if s]
    if not items:
        return []
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for sentence in items:
        extra = len(sentence) + (1 if current else 0)
        if current and current_len + extra > max_chars:
            chunks.append(" ".join(current))
            overlap = current[-overlap_sentences:] if overlap_sentences else []
            current = list(overlap)
            current_len = sum(len(s) for s in current) + max(0, len(current) - 1)
            extra = len(sentence) + (1 if current else 0)
        if len(sentence) > max_chars and not current:
            chunks.extend(char_windows(sentence, max_chars, max(0, min(50, max_chars // 10))))
            current = []
            current_len = 0
            continue
        current.append(sentence)
        current_len += extra
    if current:
        chunks.append(" ".join(current))
    return chunks


def make_chunk(
    text: str,
    *,
    document_id: str,
    language: str,
    strategy: str,
    position: int,
    metadata: dict[str, Any] | None = None,
) -> Chunk:
    chunk_id = stable_id(document_id, strategy, str(position), text[:64])
    payload = dict(metadata or {})
    return Chunk(
        text=text,
        document_id=document_id,
        chunk_id=chunk_id,
        language=language,
        chunk_strategy=strategy,
        position=position,
        metadata=payload,
    )
