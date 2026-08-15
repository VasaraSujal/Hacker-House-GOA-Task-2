"""Inspect ai4bharat/MSMARCO-XI without materializing the 55–56 GB corpus.

Streaming / Hub metadata only. Never calls list(dataset).
"""

from __future__ import annotations

import json
import sys
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATASET_ID = "ai4bharat/MSMARCO-XI"
SAMPLE_ROWS = 3
COUNT_SCAN_CAP = 250

# Config name (HF) -> filename prefix used in train/validation folders.
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


def _print_header(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def _short(value: Any, limit: int = 240) -> str:
    text = str(value).replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _summarize_value(value: Any) -> str:
    if value is None:
        return "None"
    if isinstance(value, (str, int, float, bool)):
        return f"{type(value).__name__}: {_short(value, 120)}"
    if isinstance(value, list):
        inner = type(value[0]).__name__ if value else "empty"
        preview = _short(value[0], 80) if value else ""
        return f"list(len={len(value)}, item={inner}) {preview}"
    if isinstance(value, dict):
        keys = list(value.keys())
        return f"dict(keys={keys})"
    return f"{type(value).__name__}"


def inspect_hub_files() -> dict[str, Any]:
    from huggingface_hub import HfApi, dataset_info

    api = HfApi()
    info = dataset_info(DATASET_ID)
    files = api.list_repo_files(DATASET_ID, repo_type="dataset")

    report = {
        "id": info.id,
        "sha": getattr(info, "sha", None),
        "last_modified": str(getattr(info, "last_modified", None)),
        "downloads": getattr(info, "downloads", None),
        "file_count": len(files),
        "files_by_prefix": {},
        "parquet_files": [f for f in files if f.endswith(".parquet")],
        "jsonl_files": [f for f in files if f.endswith(".jsonl")],
        "scripts": [f for f in files if f.endswith(".py")],
        "card": [f for f in files if f.lower().startswith("readme")],
    }

    prefixes: Counter[str] = Counter()
    for name in files:
        prefixes[name.split("/")[0] if "/" in name else name] += 1
    report["files_by_prefix"] = dict(prefixes)

    _print_header("Hugging Face Hub metadata")
    print(f"Dataset ID     : {report['id']}")
    print(f"Revision SHA   : {report['sha']}")
    print(f"Last modified  : {report['last_modified']}")
    print(f"File count     : {report['file_count']}")
    print(f"Top-level paths: {report['files_by_prefix']}")
    print(f"Loading scripts: {report['scripts']}")
    print(f"Parquet files  : {len(report['parquet_files'])}")
    print(f"JSONL files    : {len(report['jsonl_files'])}")
    for path in report["parquet_files"][:8]:
        print(f"  - {path}")
    if len(report["parquet_files"]) > 8:
        print(f"  ... {len(report['parquet_files']) - 8} more parquet files")
    return report


def inspect_configs() -> list[str]:
    _print_header("Dataset configurations / languages")
    configs: list[str] = []
    try:
        from datasets import get_dataset_config_names

        try:
            configs = get_dataset_config_names(DATASET_ID, trust_remote_code=True)
        except TypeError:
            configs = get_dataset_config_names(DATASET_ID)
        print(f"get_dataset_config_names: {configs}")
        if configs == ["default"] or not configs:
            print(
                "Hub parquet layout exposes a single 'default' config that concatenates "
                "all languages. Language is selected by filename (e.g. validation/hinval.parquet)."
            )
            configs = list(LANG_FILE_PREFIX.keys())
            print(f"Language configs used by this project: {configs}")
    except Exception as exc:  # noqa: BLE001
        print(f"get_dataset_config_names failed: {type(exc).__name__}: {exc}")
        print("Falling back to known MSMARCO-XI language configs.")
        configs = list(LANG_FILE_PREFIX.keys())
        print(f"Assumed configs: {configs}")

    print("\nLanguage map:")
    for code in configs:
        name = LANGUAGE_NAMES.get(code, "?")
        prefix = LANG_FILE_PREFIX.get(code, code)
        print(f"  {code:4} {name:12} files~ {prefix}train / {prefix}val")
    return configs


def inspect_splits(configs: list[str]) -> None:
    _print_header("Splits")
    try:
        from datasets import get_dataset_split_names

        probe = "hi" if "hi" in configs else (configs[0] if configs else "hi")
        try:
            splits = get_dataset_split_names(DATASET_ID, probe, trust_remote_code=True)
        except TypeError:
            splits = get_dataset_split_names(DATASET_ID, probe)
        print(f"Splits for config={probe}: {splits}")
    except Exception as exc:  # noqa: BLE001
        print(f"get_dataset_split_names failed: {type(exc).__name__}: {exc}")
        print("Assumed splits from Hub layout: train, validation")


def _stream_parquet(split: str, lang: str, max_rows: int):
    from datasets import load_dataset

    prefix = LANG_FILE_PREFIX.get(lang, lang)
    # Hub currently stores parquet (hinval.parquet), not the jsonl names in the card.
    filename = f"{prefix}{'train' if split == 'train' else 'val'}.parquet"
    data_file = f"{split}/{filename}"
    print(f"Streaming parquet: hf://datasets/{DATASET_ID}/{data_file}")
    return load_dataset(
        "parquet",
        data_files={split: f"hf://datasets/{DATASET_ID}/{data_file}"},
        split=split,
        streaming=True,
    ), data_file


def _stream_official(split: str, lang: str):
    from datasets import load_dataset

    print(f"Streaming official loader: load_dataset({DATASET_ID!r}, {lang!r}, split={split!r})")
    return load_dataset(
        DATASET_ID,
        lang,
        split=split,
        streaming=True,
        trust_remote_code=True,
    )


def inspect_schema_and_samples(lang: str = "hi", split: str = "validation") -> dict[str, Any]:
    _print_header(f"Schema + samples (config={lang}, split={split})")
    dataset = None
    source = None
    errors: list[str] = []

    try:
        dataset, source = _stream_parquet(split, lang, SAMPLE_ROWS)
        print(f"Opened via parquet streaming ({source})")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"parquet: {type(exc).__name__}: {exc}")
        print(f"Parquet streaming failed: {type(exc).__name__}: {exc}")
        try:
            dataset = _stream_official(split, lang)
            source = "official_script"
            print("Opened via official dataset script (streaming)")
        except Exception as exc2:  # noqa: BLE001
            errors.append(f"official: {type(exc2).__name__}: {exc2}")
            print("Official loader failed:")
            traceback.print_exc()
            return {"ok": False, "errors": errors}

    iterator = iter(dataset)
    samples: list[dict[str, Any]] = []
    for i in range(SAMPLE_ROWS):
        try:
            row = next(iterator)
        except StopIteration:
            break
        samples.append(row)

    if not samples:
        print("No rows returned from stream.")
        return {"ok": False, "errors": errors + ["empty stream"]}

    first = samples[0]
    print("\nTop-level fields:")
    for key, value in first.items():
        print(f"  - {key:20} {_summarize_value(value)}")

    if "passages" in first:
        passages = first["passages"]
        print("\nPassage structure:")
        if isinstance(passages, dict):
            for key, value in passages.items():
                print(f"  - passages.{key:22} {_summarize_value(value)}")
        elif isinstance(passages, list) and passages:
            print(f"  list of {type(passages[0]).__name__}, len={len(passages)}")
            if isinstance(passages[0], dict):
                print(f"  item keys: {list(passages[0].keys())}")

    print("\nRepresentative samples:")
    for idx, row in enumerate(samples, start=1):
        print(f"\n--- sample {idx} ---")
        print(f"query_id     : {row.get('query_id')}")
        print(f"query_type   : {row.get('query_type')}")
        print(f"source_lang  : {row.get('source_lang')}")
        print(f"target_lang  : {row.get('target_lang')}")
        print(f"query        : {_short(row.get('query'))}")
        print(f"Eng_Query    : {_short(row.get('Eng_Query') or row.get('eng_query'))}")
        answer = row.get("Answer", row.get("answers"))
        print(f"Answer       : {_short(answer)}")
        eng_answer = row.get("Eng_Answer", row.get("eng_answer"))
        print(f"Eng_Answer   : {_short(eng_answer)}")
        passages = row.get("passages") or {}
        if isinstance(passages, dict):
            selected = passages.get("is_selected") or []
            print(f"n_passages   : {len(passages.get('English_passages') or passages.get('Translated_passages') or [])}")
            print(f"is_selected  : {selected}")
            selected_idx = [i for i, flag in enumerate(selected) if flag]
            eng = passages.get("English_passages") or []
            tr = passages.get("Translated_passages") or []
            if selected_idx and eng:
                print(f"selected EN  : {_short(eng[selected_idx[0]])}")
            if selected_idx and tr:
                print(f"selected XI  : {_short(tr[selected_idx[0]])}")
            elif eng:
                print(f"first EN     : {_short(eng[0])}")

    _print_header("Approximate record count (capped stream scan)")
    # Re-open a fresh stream so we do not exhaust/skip past the samples above.
    try:
        count_ds, _ = _stream_parquet(split, lang, COUNT_SCAN_CAP)
    except Exception:
        count_ds = dataset
    scanned = 0
    selected_passages = 0
    passage_lens: list[int] = []
    query_types: Counter[str] = Counter()
    for row in count_ds:
        scanned += 1
        query_types[str(row.get("query_type"))] += 1
        passages = row.get("passages") or {}
        if isinstance(passages, dict):
            selected = passages.get("is_selected") or []
            selected_passages += sum(1 for flag in selected if flag)
            n = len(passages.get("English_passages") or passages.get("Translated_passages") or [])
            passage_lens.append(n)
        if scanned >= COUNT_SCAN_CAP:
            break

    avg_passages = (sum(passage_lens) / len(passage_lens)) if passage_lens else 0.0
    print(f"Scanned rows          : {scanned} (cap={COUNT_SCAN_CAP}; NOT a full split count)")
    print(f"Avg passages / row    : {avg_passages:.2f}")
    print(f"Selected passages     : {selected_passages} in scanned window")
    print(f"query_type histogram  : {dict(query_types)}")
    print("\nPublished sizes (IndicRAGSuite / dataset card, not measured here):")
    print("  ~778,638 train and ~97,941 validation examples per language.")
    print("  Full Hub snapshot is ~55.6 GB across 14 Indic languages.")
    print("  Do NOT load the full dataset into memory.")

    recommendation = {
        "config": lang,
        "split": split,
        "why": (
            "Hindi is the dataset default config and includes English originals plus "
            "Indic translations. Validation is smaller than train (~462 MB parquet vs "
            "multi-GB train) and still carries is_selected relevance labels needed for "
            "Recall@K / MRR / nDCG. Stream a MAX_DOCUMENTS subset for development."
        ),
        "dev_subset": {
            "INGEST_MODE": "subset",
            "DATASET_CONFIG": lang,
            "DATASET_SPLIT": split,
            "MAX_DOCUMENTS": 500,
            "note": (
                "Each record is a query with ~10 passages. 500 records ≈ 5,000 passage "
                "documents — a practical local prototype without touching 55 GB."
            ),
        },
    }

    _print_header("Recommended development subset")
    print(json.dumps(recommendation, indent=2))

    _print_header("Retrieval-relevant fields")
    print("Documents to index : passages.English_passages and/or passages.Translated_passages")
    print("Query fields       : query (translated), Eng_Query (English)")
    print("Relevance labels   : passages.is_selected (1 = relevant to that query)")
    print("Language           : source_lang, target_lang")
    print("Stable IDs         : query_id + passage index (passages are not globally unique IDs)")

    return {
        "ok": True,
        "source": source,
        "fields": list(first.keys()),
        "recommendation": recommendation,
        "errors": errors,
        "scanned": scanned,
    }


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    print("MSMARCO-XI inspection (streaming / metadata only)")
    print("This script will not download or materialize the full 55–56 GB dataset.")
    hub = inspect_hub_files()
    configs = inspect_configs()
    inspect_splits(configs)
    result = inspect_schema_and_samples(lang="hi", split="validation")

    _print_header("Inspection summary")
    print(f"Hub files seen     : {hub.get('file_count')}")
    print(f"Configs            : {configs}")
    print(f"Inspection ok      : {result.get('ok')}")
    print(f"Fields             : {result.get('fields')}")
    if result.get("errors"):
        print(f"Non-fatal errors   : {result['errors']}")
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
