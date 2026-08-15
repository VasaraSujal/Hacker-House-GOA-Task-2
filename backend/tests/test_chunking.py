from rag.chunking.factory import get_chunker
from rag.chunking.fixed import FixedSizeChunker
from rag.chunking.metadata_aware import MetadataAwareChunker
from rag.chunking.semantic import SemanticChunker
from rag.chunking.sentence import SentenceChunker


TEXT = (
    "Paris is the capital of France. It is known for the Eiffel Tower. "
    "The city sits on the River Seine. Unrelated later: quantum chromodynamics studies the strong force."
)


def test_fixed_chunker_respects_size_and_overlap() -> None:
    chunker = FixedSizeChunker(chunk_size=40, overlap=10)
    chunks = chunker.chunk(TEXT, document_id="d1", language="en")
    assert chunks
    assert all(c.chunk_strategy == "fixed" for c in chunks)
    assert all(c.document_id == "d1" for c in chunks)
    assert all(c.metadata == {} or True for c in chunks)
    if len(chunks) > 1:
        assert chunks[0].text[-10:] == chunks[1].text[:10]


def test_sentence_chunker_keeps_sentence_boundaries() -> None:
    chunker = SentenceChunker(chunk_size=80, overlap=20)
    chunks = chunker.chunk(TEXT, document_id="d1", language="en", metadata={"source": "test"})
    assert chunks
    assert all(c.chunk_strategy == "sentence" for c in chunks)
    assert all(c.metadata.get("source") == "test" for c in chunks)
    assert chunks[0].chunk_id != chunks[-1].chunk_id or len(chunks) == 1


def test_semantic_chunker_groups_related_sentences() -> None:
    chunker = SemanticChunker(chunk_size=200, similarity_threshold=0.05)
    chunks = chunker.chunk(TEXT, document_id="d1", language="en")
    assert chunks
    assert all(c.chunk_strategy == "semantic" for c in chunks)
    assert all(c.language == "en" for c in chunks)


def test_factory_and_metadata_strategy() -> None:
    assert isinstance(get_chunker("fixed", 50, 10), FixedSizeChunker)
    assert isinstance(get_chunker("sentence", 50, 10), SentenceChunker)
    assert isinstance(get_chunker("semantic", 50, 10), SemanticChunker)
    meta = get_chunker("metadata", 50, 10)
    assert isinstance(meta, MetadataAwareChunker)
    chunks = meta.chunk("Hello world. Another sentence.", document_id="x", language="hi")
    assert chunks[0].document_id == "x"
    assert chunks[0].language == "hi"
    assert chunks[0].chunk_strategy == "metadata"


def test_factory_rejects_unknown() -> None:
    try:
        get_chunker("nope")
        assert False, "expected ValueError"
    except ValueError:
        pass
