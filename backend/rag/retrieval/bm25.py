from __future__ import annotations

import logging
import pickle
import re
from pathlib import Path
from typing import Any, Iterable, Sequence

from rag.chunking.base import Chunk
from rag.retrieval.types import RetrievalResult

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[^\s,.;:!?।॥؟\"'()\[\]{}]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Whitespace/punctuation tokenizer that keeps Indic grapheme sequences intact."""
    return [t.lower() for t in _TOKEN_RE.findall(text or "") if t]


class BM25Index:
    """In-memory BM25 over the development subset. Persisted to disk.

    Interface is intentionally small so a later sparse engine (e.g. Qdrant
    sparse vectors) can replace this class without changing HybridRetriever.
    """

    def __init__(self) -> None:
        self._corpus_tokens: list[list[str]] = []
        self._chunks: list[dict[str, Any]] = []
        self._bm25 = None
        self._chunk_ids: set[str] = set()

    def __len__(self) -> int:
        return len(self._chunks)

    def evaluation_queries(self, limit: int | None = None) -> list[dict[str, Any]]:
        """Return deterministic queries/qrels already represented in this index."""
        grouped: dict[tuple[str, str], dict[str, Any]] = {}
        for chunk in self._chunks:
            query_id = str(chunk.get("query_id") or "")
            language = str(chunk.get("language") or "und")
            query = str(chunk.get("source_query") or "").strip()
            document_id = str(chunk.get("document_id") or "")
            if not query_id or not query or not document_id:
                continue
            key = (query_id, language)
            item = grouped.setdefault(
                key,
                {
                    "query_id": query_id,
                    "language": language,
                    "query": query,
                    "relevant_document_ids": set(),
                },
            )
            if bool(chunk.get("is_selected")):
                item["relevant_document_ids"].add(document_id)
        rows = [
            {
                **item,
                "relevant_document_ids": sorted(item["relevant_document_ids"]),
            }
            for item in grouped.values()
            if item["relevant_document_ids"]
        ]
        rows.sort(key=lambda row: (row["query_id"], row["language"]))
        return rows[:limit] if limit is not None else rows

    def chunk_statistics(self) -> dict[str, Any]:
        lengths = [len(str(chunk.get("text") or "")) for chunk in self._chunks]
        languages: dict[str, int] = {}
        for chunk in self._chunks:
            lang = str(chunk.get("language") or "und")
            languages[lang] = languages.get(lang, 0) + 1
        return {
            "chunk_count": len(lengths),
            "average_chunk_chars": (sum(lengths) / len(lengths)) if lengths else 0.0,
            "min_chunk_chars": min(lengths) if lengths else 0,
            "max_chunk_chars": max(lengths) if lengths else 0,
            "text_utf8_bytes": sum(
                len(str(chunk.get("text") or "").encode("utf-8"))
                for chunk in self._chunks
            ),
            "languages": languages,
        }

    def add_chunks(self, chunks: Sequence[Chunk]) -> int:
        added = 0
        for chunk in chunks:
            if chunk.chunk_id in self._chunk_ids:
                continue
            tokens = tokenize(chunk.text)
            if not tokens:
                continue
            self._corpus_tokens.append(tokens)
            self._chunks.append(chunk.to_payload())
            self._chunk_ids.add(chunk.chunk_id)
            added += 1
        if added:
            self._rebuild()
        return added

    def _rebuild(self) -> None:
        from rank_bm25 import BM25Okapi

        self._bm25 = BM25Okapi(self._corpus_tokens) if self._corpus_tokens else None

    def search(self, query: str, top_k: int = 20) -> list[RetrievalResult]:
        if self._bm25 is None or not self._chunks:
            return []
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        k = min(top_k, len(scores))
        if k <= 0:
            return []
        ranked = sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True)[:k]
        results: list[RetrievalResult] = []
        for i in ranked:
            payload = dict(self._chunks[int(i)])
            text = str(payload.pop("text", ""))
            document_id = str(payload.pop("document_id", ""))
            chunk_id = str(payload.pop("chunk_id", ""))
            results.append(
                RetrievalResult(
                    text=text,
                    score=float(scores[int(i)]),
                    document_id=document_id,
                    chunk_id=chunk_id,
                    metadata=payload,
                )
            )
        return results

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"corpus_tokens": self._corpus_tokens, "chunks": self._chunks}
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))
        tmp.replace(path)
        logger.info("Saved BM25 index", extra={"path": str(path), "docs": len(self._chunks)})

    @classmethod
    def load(cls, path: Path) -> "BM25Index":
        index = cls()
        if not path.exists():
            return index
        payload = pickle.loads(path.read_bytes())
        index._corpus_tokens = payload.get("corpus_tokens", [])
        index._chunks = payload.get("chunks", [])
        index._chunk_ids = {str(c.get("chunk_id")) for c in index._chunks}
        index._rebuild()
        logger.info("Loaded BM25 index", extra={"path": str(path), "docs": len(index._chunks)})
        return index
