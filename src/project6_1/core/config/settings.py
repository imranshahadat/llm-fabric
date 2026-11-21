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
    COHERE= "cohere"
    VERCEL = "vercel"



class AppSettings(_Base):
    """All LLM settings and provider credentials."""


    #------ embed selction ------


    embed_provider: LLMProviderEnum = Field(
    default=LLMProviderEnum.GEMINI, validation_alias="EMBED_PROVIDER"
    )

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
    cohere_embed_models: str = Field(default="embed-v5.0-fast", validation_alias="COHERE_EMBED_MODELS")
    cohere_rerank_models: str = Field(default="rerank-v5.0-fast", validation_alias="COHERE_RERANK_MODELS")

@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """Return a cached :class:`AppSettings` instance."""
    return AppSettings()


def reset_settings_cache() -> None:
    """Drop the cached :class:`AppSettings`. Used by tests that mutate the env."""
    get_settings.cache_clear()
