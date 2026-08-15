from rag.ingestion.cleaner import clean_text, is_usable_text, preprocess_query


def test_clean_text_strips_control_and_whitespace() -> None:
    assert clean_text("  hello\x00\n\tworld  ") == "hello world"


def test_clean_text_none() -> None:
    assert clean_text(None) == ""


def test_is_usable_text() -> None:
    assert not is_usable_text("short")
    assert is_usable_text("This passage is long enough to keep.")


def test_preprocess_query() -> None:
    assert preprocess_query("  What is MS MARCO?  ") == "What is MS MARCO?"
