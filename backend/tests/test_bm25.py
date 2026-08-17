import numpy as np

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


def test_sparse_postings_preserve_rank_bm25_scores_and_order() -> None:
    index = BM25Index()
    chunks = [
        make_chunk(
            text,
            document_id=f"d{i}",
            language="en",
            strategy="fixed",
            position=0,
        )
        for i, text in enumerate(
            [
                "A corporation issues stock to shareholders.",
                "Shareholders vote to elect a corporation's directors.",
                "Penguins live in Antarctica.",
                "A corporation is a legal entity with shareholders.",
            ]
        )
    ]
    index.add_chunks(chunks)
    tokens = tokenize("corporation shareholders shareholders")

    reference_scores = index._bm25.get_scores(tokens)
    optimized_scores = index._score_tokens(tokens)

    assert np.array_equal(optimized_scores, reference_scores)
    expected = sorted(
        range(len(reference_scores)),
        key=lambda i: float(reference_scores[i]),
        reverse=True,
    )
    assert index._top_k_indices(optimized_scores, 3) == expected[:3]


def test_partial_top_k_preserves_stable_tie_order() -> None:
    scores = np.asarray([3.0, 2.0, 2.0, 2.0, 1.0])
    assert BM25Index._top_k_indices(scores, 3) == [0, 1, 2]
