from __future__ import annotations

import logging
import pickle
import re
from collections import defaultdict
from pathlib import Path
from threading import Lock
from typing import Any, Iterable, Sequence

import numpy as np

from rag.chunking.base import Chunk
from rag.retrieval.types import RetrievalResult

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[^\s,.;:!?।॥؟\"'()\[\]{}]+", re.UNICODE)

STOPWORDS: set[str] = {
    # English stopwords & question words
    "a",
    "an",
    "the",
    "is",
    "are",
    "was",
    "were",
    "be",
    "to",
    "of",
    "in",
    "on",
    "for",
    "and",
    "or",
    "who",
    "what",
    "when",
    "where",
    "why",
    "how",
    "did",
    "does",
    "do",
    "won",
    "yesterday",
    "today",
    "tomorrow",
    "match",
    "game",
    # Hindi question words & auxiliary/grammatical particles
    "क्या",
    "है",
    "हैं",
    "था",
    "थी",
    "थे",
    "का",
    "की",
    "के",
    "में",
    "से",
    "को",
    "पर",
    "और",
    "या",
    "जो",
    "यह",
    "वह",
    "एक",
    "ने",
    "हो",
    "होता",
    "होती",
    "होते",
    "कि",
    "लिए",
    "कहाँ",
    "कहा",
    "क्यों",
    "कैसे",
    "किसने",
    "किस",
    "किसे",
    # Gujarati question words & auxiliary/grammatical particles
    "શું",
    "છે",
    "હતું",
    "હતા",
    "ના",
    "ની",
    "નું",
    "ને",
    "માં",
    "થી",
    "અને",
    "કે",
    "ક્યાં",
    "કેમ",
    "કોણ",
    "કોણે",
    "ક્યારે",
}


def tokenize(text: str) -> list[str]:
    """Whitespace/punctuation tokenizer that keeps Indic grapheme sequences intact."""
    return [t.lower() for t in _TOKEN_RE.findall(text or "") if t]


def extract_content_tokens(text_or_tokens: str | Sequence[str], min_length: int = 2) -> set[str]:
    """Return content tokens filtered of generic multilingual stopwords.

    If filtering removes all tokens, falls back to the original non-empty tokens.
    """
    if isinstance(text_or_tokens, str):
        tokens = tokenize(text_or_tokens)
    else:
        tokens = [t.lower() for t in text_or_tokens if t]
    content = {t for t in tokens if t not in STOPWORDS and len(t) >= min_length}
    if content:
        return content
    return {t for t in tokens if len(t) >= min_length} or set(tokens)


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
        self._postings: dict[str, tuple[np.ndarray, np.ndarray]] | None = None
        self._length_norm: np.ndarray | None = None
        self._postings_lock = Lock()

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
        self._postings = None
        self._length_norm = None

    def _ensure_postings(self) -> None:
        """Build a sparse query-time representation once.

        ``rank_bm25`` stores one term-frequency dictionary per document and
        scans every dictionary for every query token. The postings map keeps
        the same frequencies and BM25 parameters but visits only documents
        containing each token. It is built lazily so ingestion does not pay
        this cost after every batch rebuild.
        """
        if self._postings is not None or self._bm25 is None:
            return
        with self._postings_lock:
            if self._postings is not None or self._bm25 is None:
                return
            pending: dict[str, tuple[list[int], list[int]]] = defaultdict(
                lambda: ([], [])
            )
            for document_index, frequencies in enumerate(self._bm25.doc_freqs):
                for term, frequency in frequencies.items():
                    document_ids, term_frequencies = pending[term]
                    document_ids.append(document_index)
                    term_frequencies.append(frequency)
            self._postings = {
                term: (
                    np.asarray(document_ids, dtype=np.int32),
                    np.asarray(term_frequencies, dtype=np.int32),
                )
                for term, (document_ids, term_frequencies) in pending.items()
            }
            document_lengths = np.asarray(self._bm25.doc_len, dtype=np.float64)
            self._length_norm = self._bm25.k1 * (
                1
                - self._bm25.b
                + self._bm25.b * document_lengths / self._bm25.avgdl
            )

    def _score_tokens(self, tokens: list[str]) -> np.ndarray:
        self._ensure_postings()
        if (
            self._bm25 is None
            or self._postings is None
            or self._length_norm is None
        ):
            return np.zeros(0, dtype=np.float64)
        scores = np.zeros(self._bm25.corpus_size, dtype=np.float64)
        for term in tokens:
            posting = self._postings.get(term)
            if posting is None:
                continue
            document_ids, frequencies = posting
            scores[document_ids] += (self._bm25.idf.get(term) or 0) * (
                frequencies
                * (self._bm25.k1 + 1)
                / (frequencies + self._length_norm[document_ids])
            )
        return scores

    @staticmethod
    def _top_k_indices(scores: np.ndarray, top_k: int) -> list[int]:
        """Return exact score-descending/index-ascending top-k without full sort."""
        k = min(top_k, len(scores))
        if k <= 0:
            return []
        if k == len(scores):
            candidates = np.arange(len(scores), dtype=np.intp)
        else:
            threshold = np.partition(scores, len(scores) - k)[len(scores) - k]
            above = np.flatnonzero(scores > threshold)
            tied = np.flatnonzero(scores == threshold)[: k - len(above)]
            candidates = np.concatenate((above, tied))
        order = np.lexsort((candidates, -scores[candidates]))
        return [int(index) for index in candidates[order]]

    def search(self, query: str, top_k: int = 20) -> list[RetrievalResult]:
        if self._bm25 is None or not self._chunks:
            return []
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self._score_tokens(tokens)
        ranked = self._top_k_indices(scores, top_k)
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
