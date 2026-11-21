"""Factory for creating LLM provider instances based on application settings."""

from __future__ import annotations

from ....core.config.settings import AppSettings, get_settings
from ....core.exceptions.llm import LLMProviderError

# from app.domains.infrastructure.llm.openai_chat import OpenAILLM
# from app.domains.infrastructure.llm.openrouter_chat import OpenRouterLLM
# from app.domains.infrastructure.llm.vercel_chat import VercelLLM
from ...interfaces.llm.base import BaseLLM
from ._builder import build_provider

# from app.domains.infrastructure.llm.fake_chat import FakeLLM
from .gemini_chat import GeminiChatModel

# Provider name -> adapter class. Keys MUST cover every literal in
# :class:`LLMProviderEnum` so the config can never ask for a provider we
# can't build.
LLM_REGISTRY: dict[str, type[BaseLLM]] = {
    # "fake": FakeLLM,
    # "openai": OpenAILLM,
    # "openrouter": OpenRouterLLM,
    "gemini": GeminiChatModel,
    # "vercel": VercelLLM,
}

# Provider name -> dotted path into AppSettings for the required API key.
# Providers that need NO key (e.g. the fake adapter) are absent from this map.
API_KEY_REQUIREMENTS: dict[str, str] = {
    "openai": "openai_api_key",
    "openrouter": "openrouter_api_key",
    "gemini": "gemini_api_key",
    "vercel": "vercel_api_key",
}


def build_llm(settings: AppSettings | None = None) -> BaseLLM:
    """Create and return the configured LLM provider instance."""
    s = settings or get_settings()
    provider = s.llm_provider.value.lower()
    return build_provider(
        registry=LLM_REGISTRY,
        provider=provider,
        kind="LLM",
        api_key_path=API_KEY_REQUIREMENTS.get(provider),
        error=LLMProviderError,
        settings=s,
    )


def available_providers() -> tuple[str, ...]:
    """Return a tuple of all registered LLM provider names."""
    return tuple(LLM_REGISTRY.keys())
