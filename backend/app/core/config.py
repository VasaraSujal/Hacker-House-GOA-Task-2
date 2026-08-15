from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BACKEND_DIR.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BACKEND_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_env: str = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    dataset_id: str = "ai4bharat/MSMARCO-XI"
    dataset_config: str = "hi"
    dataset_split: str = "validation"
    index_english: bool = True
    index_translated: bool = True

    ingest_mode: Literal["subset", "full"] = "subset"
    max_documents: int = 500
    batch_size: int = 64
    checkpoint_path: Path = PROJECT_ROOT / "data" / "checkpoints" / "ingest.json"
    ingest_max_batch_retries: int = 2
    ingest_checkpoint_id_window: int = 50000
    ingest_stop_on_batch_failure: bool = True

    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "hh_goa_rag"
    qdrant_api_key: str | None = None
    qdrant_timeout_s: float = 30.0

    voice_rate_limit_per_minute: int = 20

    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embedding_normalize: bool = True
    embedding_batch_size: int = 64
    embedding_device: str = "cpu"
    embedding_eval_models: str = (
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2,"
        "intfloat/multilingual-e5-small"
    )

    reranker_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    reranker_device: str = "cpu"
    enable_reranker: bool = False

    dense_top_k: int = 20
    bm25_top_k: int = 20
    hybrid_top_k: int = 20
    rerank_top_k: int = 5
    fusion_method: Literal["rrf", "weighted"] = "rrf"
    rrf_k: int = 60
    dense_weight: float = 0.6
    bm25_weight: float = 0.4
    parallel_retrieval: bool = True

    chunk_strategy: Literal["fixed", "sentence", "semantic", "metadata"] = "sentence"
    chunk_size: int = 500
    chunk_overlap: int = 50
    semantic_similarity_threshold: float = 0.35

    max_context_chars: int = 6000
    max_query_chars: int = 2000
    relevance_min_results: int = 1
    relevance_min_score: float = 0.01
    grounding_min_overlap: float = 0.20

    bm25_index_path: Path = PROJECT_ROOT / "data" / "indexes" / "bm25.pkl"

    elevenlabs_api_key: str = ""
    elevenlabs_api_base: str = "https://api.elevenlabs.io"
    elevenlabs_model: str = "gemini-2.0-flash"
    elevenlabs_agent_id: str = ""
    elevenlabs_timeout_s: float = 30.0
    elevenlabs_max_retries: int = 3
    elevenlabs_stt_model: str = "scribe_v2"
    max_audio_size_mb: int = 20
    elevenlabs_stt_timeout_s: float = 30.0
    elevenlabs_stt_max_retries: int = 3
    voice_benchmark_queries: int = 20

    benchmark_queries: int = 100
    benchmark_warmup_queries: int = 3
    benchmark_include_generation: bool = True
    benchmark_output_dir: Path = PROJECT_ROOT / "benchmarks"
    eval_queries: int = 50

    refusal_message: str = (
        "I couldn't find enough relevant information in the provided "
        "knowledge base to answer this question."
    )

    @field_validator("checkpoint_path", "bm25_index_path", "benchmark_output_dir", mode="before")
    @classmethod
    def _resolve_path(cls, value: str | Path) -> Path:
        path = Path(value)
        if not path.is_absolute():
            path = (BACKEND_DIR / path).resolve()
        return path

    @property
    def project_root(self) -> Path:
        return PROJECT_ROOT

    @property
    def data_dir(self) -> Path:
        return PROJECT_ROOT / "data"

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
