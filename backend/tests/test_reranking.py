from rag.reranking.cross_encoder import CrossEncoderReranker
from rag.retrieval.types import RetrievalResult


class _Predict:
    def predict(self, pairs, show_progress_bar=False):
        return [0.1, 0.9]


def test_cross_encoder_orders_by_score(monkeypatch) -> None:
    reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
    reranker.model_name = "fake"
    reranker.device = "cpu"
    reranker._model = _Predict()
    c1 = RetrievalResult("wrong", 0.5, "d1", "c1")
    c2 = RetrievalResult("right", 0.4, "d2", "c2")
    out = reranker.rerank("q", [c1, c2], top_k=1)
    assert len(out) == 1
    assert out[0].chunk_id == "c2"
    assert out[0].score == 0.9
