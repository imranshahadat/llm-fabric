"""App settings for LLM chat completions loaded from .env."""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# config.py is at src/project6_1/core/config/settings.py
# -> parents[3] == project6_1_llm/  (the project root with .env)
BASE_DIR = Path(__file__).resolve().parents[3]
ENV_FILE = BASE_DIR / ".env"


class _Base(BaseSettings):
    """Base settings class that loads variables from .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


class LLMProviderEnum(StrEnum):
    """The set of providers the LLM factory can build."""

    FAKE = "fake"
    OPENAI = "openai"
    OPENROUTER = "openrouter"
    GEMINI = "gemini"
    COHERE = "cohere"
    VERCEL = "vercel"


class AppSettings(_Base):
    """All LLM settings and provider credentials."""

    # ---- Reranker selection ----
    rerank_provider: LLMProviderEnum = Field(
        default=LLMProviderEnum.COHERE, validation_alias="RERANKER_PROVIDER"
    )
    rerank_top_k: int = Field(default=4, validation_alias="RERANK_TOP_K")

    # ------ embed selction ------

    embed_provider: LLMProviderEnum = Field(
        default=LLMProviderEnum.GEMINI, validation_alias="EMBED_PROVIDER"
    )

    # ---- Vector Database Selection ----
    vector_store_provider: str = Field(default="qdrant", validation_alias="VECTOR_STORE_PROVIDER")

    # ---- LLM selection ----
    llm_provider: LLMProviderEnum = Field(
        default=LLMProviderEnum.GEMINI, validation_alias="LLM_PROVIDER"
    )
    llm_model: str = Field(default="gpt-4o-mini", validation_alias="LLM_MODEL")
    llm_temperature: float = Field(default=0.0, validation_alias="LLM_TEMPERATURE")
    llm_max_tokens: int | None = Field(default=None, validation_alias="LLM_MAX_TOKENS")
    llm_timeout: float = Field(default=30.0, validation_alias="LLM_TIMEOUT")
    llm_max_retries: int = Field(default=3, validation_alias="LLM_MAX_RETRIES")
    llm_announce: bool = Field(default=True, validation_alias="LLM_ANNOUNCE")

    # ---- OpenAI ----
    openai_api_key: SecretStr | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    openai_base_url: str = Field(
        default="https://api.openai.com/v1", validation_alias="OPENAI_BASE_URL"
    )
    openai_organization: str | None = Field(default=None, validation_alias="OPENAI_ORGANIZATION")

    # ---- OpenRouter (OpenAI-compatible gateway) ----
    openrouter_api_key: SecretStr | None = Field(
        default=None, validation_alias="OPENROUTER_API_KEY"
    )
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1", validation_alias="OPENROUTER_BASE_URL"
    )
    openrouter_model: str = Field(default="openai/gpt-4o-mini", validation_alias="OPENROUTER_MODEL")
    openrouter_site_url: str = Field(
        default="https://example.com", validation_alias="OPENROUTER_SITE_URL"
    )
    openrouter_app_name: str = Field(default="project6-1", validation_alias="OPENROUTER_APP_NAME")
    openrouter_embedding_model: str = Field(
        default="openai/text-embedding-3-small",
        validation_alias="OPENROUTER_EMBED_MODEL",
    )

    # ---- Google Gemini ----
    gemini_api_key: SecretStr | None = Field(default=None, validation_alias="GEMINI_API_KEY")
    gemini_model: str = Field(default="gemini-1.5-flash", validation_alias="GEMINI_MODEL")
    gemini_embedding_model: str = Field(
        default="gemini-embedding-001", validation_alias="GEMINI_EMBEDDING_MODEL"
    )

    # ---- Vercel AI Gateway (OpenAI-compatible) ----
    vercel_api_key: SecretStr | None = Field(default=None, validation_alias="VERCEL_API_KEY")
    vercel_base_url: str = Field(
        default="https://ai-gateway.vercel.sh/v1", validation_alias="VERCEL_BASE_URL"
    )
    vercel_model: str = Field(default="openai/gpt-4o-mini", validation_alias="VERCEL_MODEL")
    vercel_embedding_model: str = Field(
        default="text-embedding-3-small", validation_alias="VERCEL_EMBEDDING_MODEL"
    )
    # ----- Cohere -----
    cohere_api_key: SecretStr | None = Field(default=None, validation_alias="COHERE_API_KEY")
    co_api_url: str = Field(default="https://api.cohere.com/v2", validation_alias="CO_API_URL")
    cohere_embed_models: str = Field(
        default="embed-v5.0-fast", validation_alias="COHERE_EMBED_MODELS"
    )
    cohere_rerank_models: str = Field(
        default="rerank-v5.0-fast", validation_alias="COHERE_RERANK_MODELS"
    )

    # ---- Qdrant ----

    # Vector Store
    qdrant_url: str = Field(default="http://localhost:6333", validation_alias="QDRANT_URL")
    qdrant_api_key: SecretStr | None = Field(default=None, validation_alias="QDRANT_API_KEY")
    qdrant_collection_name: str = Field(
        default="rag-knowledge-base", validation_alias="QDRANT_COLLECTION"
    )
    qdrant_vector_size: int = Field(default=1024, validation_alias="QDRANT_DIMENSION")

    # Hybrid Search
    use_hybrid_search: bool = Field(default=True, validation_alias="USE_HYBRID_SEARCH")
    hybrid_fetch_k: int = Field(default=20, validation_alias="HYBRID_FETCH_K")

    # ---- Pinecone (alternative) ----
    pinecone_api_key: SecretStr | None = Field(default=None, validation_alias="PINECONE_API_KEY")
    pinecone_index_name: str = Field(
        default="rag-knowledge-base", validation_alias="PINECONE_INDEX_NAME"
    )

    # ---- Ingestion (upload / load / split) ----
    ingest_upload_dir: str = Field(default="uploads", validation_alias="INGEST_UPLOAD_DIR")
    ingest_max_file_mb: int = Field(default=100, validation_alias="INGEST_MAX_FILE_MB")
    ingest_chunk_size: int = Field(default=1200, validation_alias="INGEST_CHUNK_SIZE")
    ingest_chunk_overlap: int = Field(default=200, validation_alias="INGEST_CHUNK_OVERLAP")
    ingest_enable_ocr: bool = Field(default=True, validation_alias="INGEST_ENABLE_OCR")
    ingest_ocr_language: str = Field(default="eng", validation_alias="INGEST_OCR_LANGUAGE")
    ingest_ocr_dpi: int = Field(default=200, validation_alias="INGEST_OCR_DPI")
    ingest_pdf_extract_images: bool = Field(
        default=True, validation_alias="INGEST_PDF_EXTRACT_IMAGES"
    )

    # ---- Semantic cache (Redis / Redis Cloud) ----
    cache_enabled: bool = Field(default=True, validation_alias="CACHE_ENABLED")
    cache_provider: str = Field(default="redis", validation_alias="CACHE_PROVIDER")
    redis_url: str = Field(default="redis://localhost:6379/0", validation_alias="REDIS_URL")
    cache_namespace: str = Field(default="semcache", validation_alias="CACHE_NAMESPACE")
    cache_ttl_seconds: int = Field(default=3600, validation_alias="CACHE_TTL_SECONDS")
    cache_similarity_threshold: float = Field(
        default=0.92, validation_alias="CACHE_SIMILARITY_THRESHOLD"
    )
    cache_max_entries_per_scope: int = Field(
        default=128, validation_alias="CACHE_MAX_ENTRIES_PER_SCOPE"
    )

    # ---- Metadata filtering ----
    # Comma-separated allowlist of payload fields callers may filter on.
    metadata_filter_fields: str = Field(
        default="source,file_type,kind,page", validation_alias="METADATA_FILTER_FIELDS"
    )

    # ---- Query summarizer (small LLM that condenses long queries) ----
    query_summarizer_enabled: bool = Field(
        default=False, validation_alias="QUERY_SUMMARIZER_ENABLED"
    )
    summarizer_model: str = Field(
        default="", validation_alias="SUMMARIZER_MODEL"
    )  # empty = reuse LLM_MODEL
    summarizer_max_tokens: int = Field(default=120, validation_alias="SUMMARIZER_MAX_TOKENS")
    summarizer_min_query_words: int = Field(
        default=12, validation_alias="SUMMARIZER_MIN_QUERY_WORDS"
    )
    summarizer_max_words: int = Field(default=30, validation_alias="SUMMARIZER_MAX_WORDS")

    @property
    def allowed_metadata_filter_fields(self) -> frozenset[str]:
        """Return the parsed metadata filter allowlist (empty set = filtering off)."""
        return frozenset(
            field.strip() for field in self.metadata_filter_fields.split(",") if field.strip()
        )


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """Return a cached :class:`AppSettings` instance."""
    return AppSettings()


def reset_settings_cache() -> None:
    """Drop the cached :class:`AppSettings`. Used by tests that mutate the env."""
    get_settings.cache_clear()
