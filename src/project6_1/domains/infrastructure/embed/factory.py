"""Factory for creating embedding provider instances based on application settings."""

from __future__ import annotations

from collections.abc import Callable

from ....core.config.settings import AppSettings, get_settings
from ....core.exceptions.llm import LLMProviderError
from ...interfaces.embed.base import BaseEmbed
from ._builder import build_provider_adapter
from .gemini_embed import GeminiEmbeddingModel

# Provider name -> adapter class. Keys MUST cover every literal in
# the embedding provider enum so the config can never ask for a provider we
# can't build.
EMBED_REGISTRY: dict[str, Callable[[AppSettings], BaseEmbed]] = {
    "gemini": GeminiEmbeddingModel,
}

# Provider name -> dotted path into AppSettings for the required API key.
# Providers that need NO key are absent from this map.
EMBED_API_KEY_REQUIREMENTS: dict[str, str] = {
    "gemini": "gemini_api_key",
}


def build_embed(settings: AppSettings | None = None) -> BaseEmbed:
    """Create and return the configured embedding provider instance."""
    s = settings or get_settings()
    provider = s.embed_provider.value.lower()
    return build_provider_adapter(
        provider_registry=EMBED_REGISTRY,
        provider_name=provider,
        provider_kind="embedding",
        api_key_setting_path=EMBED_API_KEY_REQUIREMENTS.get(provider),
        unknown_provider_error=LLMProviderError,
        application_settings=s,
    )


def available_embed_providers() -> tuple[str, ...]:
    """Return a tuple of all registered embedding provider names."""
    return tuple(EMBED_REGISTRY.keys())
