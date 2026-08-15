from rag.ingestion.dataset_loader import parse_record


def _row() -> dict:
    return {
        "query_id": 42,
        "query": "पेरिस की राजधानी क्या है?",
        "Eng_Query": "What is the capital of France?",
        "Answer": "पेरिस",
        "Eng_Answer": "Paris",
        "query_type": "LOCATION",
        "source_lang": "eng_Latn",
        "target_lang": "hin_Deva",
        "passages": {
            "is_selected": [1, 0],
            "English_passages": [
                "Paris is the capital and most populous city of France.",
                "Berlin is the capital of Germany.",
            ],
            "Translated_passages": [
                "पेरिस फ्रांस की राजधानी और सबसे अधिक आबादी वाला शहर है।",
                "बर्लिन जर्मनी की राजधानी है।",
            ],
        },
    }


def test_parse_record_extracts_english_and_translated() -> None:
    record = parse_record(_row(), index_english=True, index_translated=True)
    assert record.query_id == "42"
    assert "capital of France" in record.english_query
    assert len(record.passages) == 4
    selected = [p for p in record.passages if p.is_selected]
    assert len(selected) == 2
    langs = {p.language for p in record.passages}
    assert "en" in langs
    assert "hin" in langs


def test_parse_record_can_index_english_only() -> None:
    record = parse_record(_row(), index_english=True, index_translated=False)
    assert len(record.passages) == 2
    assert all(p.language == "en" for p in record.passages)
