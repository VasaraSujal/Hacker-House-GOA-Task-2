from pathlib import Path

from app.core.config import Settings
from app.core.exceptions import EmbeddingError
from rag.chunking.metadata import make_chunk
from rag.embeddings.base import EmbeddingProvider
from rag.ingestion.indexer import Indexer
from rag.retrieval.bm25 import BM25Index
from rag.retrieval.qdrant_store import QdrantStore
from rag.ingestion.dataset_loader import QueryRecord
from rag.ingestion.cleaner import Passage
import numpy as np


class FakeEmbeddings(EmbeddingProvider):
    def __init__(self, fail_times: int = 0) -> None:
        self.fail_times = fail_times
        self.calls = 0

    @property
    def dimension(self) -> int:
        return 4

    @property
    def model_name(self) -> str:
        return "fake-embeddings"

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        self.calls += 1
        if self.calls <= self.fail_times:
            raise EmbeddingError("transient embedding failure")
        return np.zeros((len(texts), 4), dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return np.zeros(4, dtype=np.float32)


class FakeStore:
    def __init__(self) -> None:
        self.points: list[str] = []

    def ensure_collection(self) -> None:
        return None

    def upsert_chunks(self, chunks, vectors) -> int:
        for chunk in chunks:
            self.points.append(chunk.chunk_id)
        return len(chunks)

    def ping(self) -> bool:
        return True

    def count(self) -> int:
        return len(self.points)


def test_indexer_retries_transient_batch_failure(tmp_path: Path, monkeypatch) -> None:
    settings = Settings(
        ingest_mode="subset",
        max_documents=1,
        batch_size=1,
        checkpoint_path=tmp_path / "ckpt.json",
        bm25_index_path=tmp_path / "bm25.pkl",
        ingest_max_batch_retries=2,
    )
    record = QueryRecord(
        query_id="q1",
        query="What is Paris?",
        english_query="What is Paris?",
        answer="Paris",
        english_answer="Paris",
        query_type="LOCATION",
        source_lang="eng_Latn",
        target_lang="hin_Deva",
        passages=[
            Passage(
                document_id="d1",
                query_id="q1",
                passage_index=0,
                text="Paris is the capital of France.",
                language="en",
                is_selected=True,
                source_query="What is Paris?",
                metadata={},
            )
        ],
    )

    def fake_iter_records(**kwargs):
        yield record

    monkeypatch.setattr("rag.ingestion.indexer.iter_records", fake_iter_records)
    embeddings = FakeEmbeddings(fail_times=1)
    store = FakeStore()
    indexer = Indexer(settings, embeddings, store, BM25Index(), max_batch_retries=2)
    stats = indexer.ingest()
    assert stats.chunks >= 1
    assert stats.retries >= 1
    assert embeddings.calls >= 2
    assert store.count() >= 1


def test_bm25_skips_duplicate_chunk_ids() -> None:
    index = BM25Index()
    chunk = make_chunk(
        "Paris is the capital of France.",
        document_id="d1",
        language="en",
        strategy="sentence",
        position=0,
    )
    assert index.add_chunks([chunk]) == 1
    assert index.add_chunks([chunk]) == 0
    assert len(index) == 1
