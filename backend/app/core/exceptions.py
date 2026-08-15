from __future__ import annotations


class RAGError(Exception):
    def __init__(self, message: str, status_code: int = 500, code: str = "rag_error") -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.code = code


class InvalidQueryError(RAGError):
    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=400, code="invalid_query")


class ConfigurationError(RAGError):
    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=500, code="invalid_configuration")


class DatasetError(RAGError):
    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=500, code="dataset_error")


class VectorStoreError(RAGError):
    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=503, code="vector_store_unavailable")


class EmbeddingError(RAGError):
    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=500, code="embedding_error")


class GenerationError(RAGError):
    def __init__(self, message: str, status_code: int = 502) -> None:
        super().__init__(message, status_code=status_code, code="generation_error")


class STTError(RAGError):
    def __init__(self, message: str, status_code: int = 502, code: str = "stt_error") -> None:
        super().__init__(message, status_code=status_code, code=code)


class InvalidAudioError(STTError):
    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=400, code="invalid_audio")


class RerankerError(RAGError):
    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=500, code="reranker_error")


class RetrievalEmptyError(RAGError):
    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=200, code="empty_retrieval")
