from tests.conftest import FakeEmbeddings


def test_fake_embeddings_dimension_and_normalization() -> None:
    emb = FakeEmbeddings(dim=8)
    assert emb.dimension == 8
    docs = emb.embed_documents(["alpha", "beta"])
    assert docs.shape == (2, 8)
    q = emb.embed_query("alpha")
    assert q.shape == (8,)
    # same text is deterministic
    assert abs(float((q * docs[0]).sum()) - 1.0) < 1e-5
