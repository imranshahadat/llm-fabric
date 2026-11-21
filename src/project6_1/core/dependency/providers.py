"""Creates and caches the active LLM client for the application."""

from __future__ import annotations

from functools import lru_cache

from project6_1.core.config.settings import (
    LLMProviderEnum,
    get_settings,
    reset_settings_cache,
)

from ...domains.infrastructure.llm.factory import build_llm
from ...domains.interfaces.llm.base import BaseLLM
from ..logging.setup import get_logger

# Per-provider default base URLs. Providers that don't use an HTTP base URL
# (the in-process ``fake`` adapter, Google Gemini via the google-generativeai
# SDK) map to the empty string. These mirror the per-field defaults on
# ``AppSettings`` so there is a single source of truth for each value.
DEFAULT_BASE_URLS: dict[LLMProviderEnum, str] = {
    LLMProviderEnum.FAKE: "",
    LLMProviderEnum.OPENAI: "https://api.openai.com/v1",
    LLMProviderEnum.OPENROUTER: "https://openrouter.ai/api/v1",
    LLMProviderEnum.GEMINI: "",
    LLMProviderEnum.VERCEL: "https://ai-gateway.vercel.sh/v1",
    LLMProviderEnum.COHERE: "https://api.cohere.com/v2",
}

# Per-provider default model identifiers; same values as the corresponding
# ``*_model`` field defaults on ``AppSettings``.
DEFAULT_MODELS: dict[LLMProviderEnum, str] = {
    LLMProviderEnum.FAKE: "fake",
    LLMProviderEnum.OPENAI: "gpt-4o-mini",
    LLMProviderEnum.OPENROUTER: "openai/gpt-4o-mini",
    LLMProviderEnum.GEMINI: "gemini-1.5-flash",
    LLMProviderEnum.VERCEL: "openai/gpt-4o-mini",
    LLMProviderEnum.COHERE: "command-r-plus",
}


__all__ = [
    "DEFAULT_BASE_URLS",
    "DEFAULT_MODELS",
    "LLMProviderEnum",
    "get_llm",
    "reset_llm_cache",
]


@lru_cache(maxsize=1)
def get_llm() -> BaseLLM:
    """Return a cached BaseLLM instance, initializing it on the first call."""
    settings = get_settings()
    llm = build_llm(settings)
    if settings.llm_announce:
        get_logger().info(
            "llm_wired",
            provider=settings.llm_provider.value,
            model=settings.llm_model or "(default)",
        )
    return llm


def reset_llm_cache() -> None:
    """Clear the cached LLM client and settings so they rebuild on the next call."""
    get_llm.cache_clear()
    reset_settings_cache()
