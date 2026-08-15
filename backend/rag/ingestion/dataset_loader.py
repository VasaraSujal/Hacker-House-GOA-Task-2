from __future__ import annotations

import hashlib
import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from app.core.exceptions import DatasetError
from rag.ingestion.cleaner import Passage, clean_text, is_usable_text

logger = logging.getLogger(__name__)

DATASET_ID_DEFAULT = "ai4bharat/MSMARCO-XI"

LANG_FILE_PREFIX = {
    "as": "asm",
    "bn": "ben",
    "gu": "guj",
    "hi": "hin",
    "kn": "kan",
    "ml": "mal",
    "mr": "mar",
    "ne": "nep",
    "or": "ori",
    "pa": "pan",
    "sa": "san",
    "ta": "tam",
    "te": "tel",
    "ur": "urd",
}

LANGUAGE_NAMES = {
    "as": "Assamese",
    "bn": "Bengali",
    "gu": "Gujarati",
    "hi": "Hindi",
    "kn": "Kannada",
    "ml": "Malayalam",
    "mr": "Marathi",
    "ne": "Nepali",
    "or": "Odia",
    "pa": "Punjabi",
    "sa": "Sanskrit",
    "ta": "Tamil",
    "te": "Telugu",
    "ur": "Urdu",
}


@dataclass(slots=True)
class QueryRecord:
    query_id: str
    query: str
    english_query: str
    answer: str
    english_answer: str
    query_type: str
    source_lang: str
    target_lang: str
    passages: list[Passage] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def _passage_id(query_id: str, index: int, language: str, text: str) -> str:
    digest = hashlib.sha1(f"{query_id}|{index}|{language}|{text[:80]}".encode("utf-8")).hexdigest()[:16]
    return f"{query_id}-{language}-{index}-{digest}"


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return list(value)


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return int(value) == 1
    if isinstance(value, str):
        return value.strip() in {"1", "true", "True", "yes"}
    return bool(value)


def _parquet_path(dataset_id: str, split: str, config: str) -> str:
    prefix = LANG_FILE_PREFIX.get(config, config)
    filename = f"{prefix}{'train' if split == 'train' else 'val'}.parquet"
    return f"{split}/{filename}"


def open_streaming_dataset(
    *,
    dataset_id: str = DATASET_ID_DEFAULT,
    config: str = "hi",
    split: str = "validation",
):
    """Open MSMARCO-XI as a streaming IterableDataset. Never materializes the split."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise DatasetError("The `datasets` package is required for ingestion") from exc

    data_file = _parquet_path(dataset_id, split, config)
    uri = f"hf://datasets/{dataset_id}/{data_file}"
    logger.info("Opening streaming dataset", extra={"uri": uri, "config": config, "split": split})
    try:
        return load_dataset("parquet", data_files={split: uri}, split=split, streaming=True)
    except Exception as parquet_exc:  # noqa: BLE001
        logger.warning("Parquet streaming failed; trying official loader", extra={"error": str(parquet_exc)})
        try:
            try:
                return load_dataset(dataset_id, config, split=split, streaming=True, trust_remote_code=True)
            except TypeError:
                return load_dataset(dataset_id, config, split=split, streaming=True)
        except Exception as exc:  # noqa: BLE001
            raise DatasetError(
                f"Failed to load {dataset_id} config={config} split={split}: {exc}"
            ) from exc


def parse_record(
    row: dict[str, Any],
    *,
    index_english: bool = True,
    index_translated: bool = True,
) -> QueryRecord:
    query_id = str(row.get("query_id", ""))
    query = clean_text(row.get("query"))
    english_query = clean_text(row.get("Eng_Query") or row.get("eng_query"))
    answer = row.get("Answer", row.get("answers"))
    if isinstance(answer, list):
        answer = " ".join(str(a) for a in answer)
    answer = clean_text(answer)
    english_answer = clean_text(row.get("Eng_Answer") or row.get("eng_answer"))
    source_lang = str(row.get("source_lang") or "eng_Latn")
    target_lang = str(row.get("target_lang") or "")
    query_type = str(row.get("query_type") or "")

    raw_passages = row.get("passages") or {}
    selected = _as_list(raw_passages.get("is_selected")) if isinstance(raw_passages, dict) else []
    english_passages = _as_list(raw_passages.get("English_passages")) if isinstance(raw_passages, dict) else []
    translated_passages = _as_list(raw_passages.get("Translated_passages")) if isinstance(raw_passages, dict) else []
    n = max(len(english_passages), len(translated_passages), len(selected))

    passages: list[Passage] = []
    for i in range(n):
        is_sel = _truthy(selected[i]) if i < len(selected) else False
        if index_english and i < len(english_passages):
            text = clean_text(english_passages[i])
            if is_usable_text(text):
                passages.append(
                    Passage(
                        document_id=_passage_id(query_id, i, "en", text),
                        query_id=query_id,
                        passage_index=i,
                        text=text,
                        language="en",
                        is_selected=is_sel,
                        source_query=english_query or query,
                        metadata={"source_lang": source_lang, "field": "English_passages"},
                    )
                )
        if index_translated and i < len(translated_passages):
            text = clean_text(translated_passages[i])
            if is_usable_text(text):
                lang = target_lang.split("_")[0] if target_lang else "und"
                passages.append(
                    Passage(
                        document_id=_passage_id(query_id, i, lang, text),
                        query_id=query_id,
                        passage_index=i,
                        text=text,
                        language=lang,
                        is_selected=is_sel,
                        source_query=query or english_query,
                        metadata={"target_lang": target_lang, "field": "Translated_passages"},
                    )
                )

    return QueryRecord(
        query_id=query_id,
        query=query,
        english_query=english_query,
        answer=answer,
        english_answer=english_answer,
        query_type=query_type,
        source_lang=source_lang,
        target_lang=target_lang,
        passages=passages,
        metadata={"query_type": query_type},
    )


def iter_records(
    *,
    dataset_id: str = DATASET_ID_DEFAULT,
    config: str = "hi",
    split: str = "validation",
    max_documents: int | None = 500,
    ingest_mode: str = "subset",
    index_english: bool = True,
    index_translated: bool = True,
    skip_query_ids: set[str] | None = None,
) -> Iterator[QueryRecord]:
    """Yield query records from a streaming dataset.

    max_documents counts *query records*, not unique passages. Use ingest_mode=full
    to stream an entire split without a cap (still not loaded into RAM).
    """
    if ingest_mode not in {"subset", "full"}:
        raise DatasetError(f"Invalid INGEST_MODE={ingest_mode}")
    dataset = open_streaming_dataset(dataset_id=dataset_id, config=config, split=split)
    seen = skip_query_ids or set()
    emitted = 0
    for row in dataset:
        record = parse_record(row, index_english=index_english, index_translated=index_translated)
        if not record.query_id or record.query_id in seen:
            continue
        yield record
        emitted += 1
        if ingest_mode == "subset" and max_documents is not None and emitted >= max_documents:
            break


def iter_passages(records: Iterator[QueryRecord]) -> Iterator[Passage]:
    for record in records:
        yield from record.passages
