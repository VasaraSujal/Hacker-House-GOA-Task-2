from rag.context.builder import ContextBuilder
from rag.retrieval.types import RetrievalResult


def test_context_builder_dedupes_and_limits() -> None:
    results = [
        RetrievalResult("Paris is the capital of France.", 0.9, "d1", "c1"),
        RetrievalResult("Paris is the capital of France.", 0.8, "d1", "c1-dup"),
        RetrievalResult("Berlin is in Germany.", 0.7, "d2", "c2"),
        RetrievalResult("Madrid is in Spain.", 0.6, "d3", "c3"),
    ]
    builder = ContextBuilder(max_chars=180)
    context, selected = builder.build(results)
    assert "Source 1" in context
    assert len(selected) >= 1
    texts = [" ".join(s.text.split()).lower() for s in selected]
    assert len(texts) == len(set(texts))
    assert all("Document ID:" in context for _ in [0])
