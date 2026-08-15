from rag.chunking.metadata import make_chunk
from rag.retrieval.bm25 import BM25Index, tokenize


def test_tokenize_unicode() -> None:
    tokens = tokenize("Paris is great. पेरिस")
    assert "paris" in tokens
    assert "पेरिस" in tokens


def test_bm25_ranks_relevant_chunk_first() -> None:
    index = BM25Index()
    a = make_chunk(
        "The Eiffel Tower is located in Paris, the capital of France.",
        document_id="d1",
        language="en",
        strategy="fixed",
        position=0,
    )
    b = make_chunk(
        "Penguins live in Antarctica and hunt for fish.",
        document_id="d2",
        language="en",
        strategy="fixed",
        position=0,
    )
    index.add_chunks([a, b])
    hits = index.search("Eiffel Tower Paris France", top_k=2)
    assert hits
    assert hits[0].document_id == "d1"
    assert hits[0].chunk_id == a.chunk_id
