from rag.embeddings.base import EmbeddingProvider
from rag.embeddings.cloud_meta import CloudInferenceMetaEmbeddings

__all__ = ["EmbeddingProvider", "CloudInferenceMetaEmbeddings", "LocalEmbeddingProvider"]


def __getattr__(name: str):
    """Keep the local provider import lazy for the memory-constrained image."""
    if name == "LocalEmbeddingProvider":
        from rag.embeddings.local import LocalEmbeddingProvider

        return LocalEmbeddingProvider
    raise AttributeError(name)
