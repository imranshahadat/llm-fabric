"""Factory for creating vector store provider instances."""

from __future__ import annotations

from ....core.config.settings import AppSettings, get_settings
from ....core.exceptions.llm import LLMProviderError
from ...interfaces.vector_store.base import BaseVectorStore
from ._builder import build_provider_adapter
from .qdrant_store import QdrantVectorStore

VECTOR_STORE_REGISTRY: dict[str, type[BaseVectorStore]] = {
    "qdrant": QdrantVectorStore,
}

API_KEY_REQUIREMENTS: dict[str, str] = {
    "qdrant": "qdrant_api_key",
}


def build_vector_store(settings: AppSettings | None = None) -> BaseVectorStore:
    """Create and return the configured vector store provider instance."""
    s = settings or get_settings()
    provider = s.vector_store_provider.lower()

    return build_provider_adapter(
        provider_registry=VECTOR_STORE_REGISTRY,
        provider_name=provider,
        provider_kind="vector store",
        api_key_setting_path=API_KEY_REQUIREMENTS.get(provider),
        application_settings=s,
        unknown_provider_error=LLMProviderError,
    )


def available_vector_store_providers() -> tuple[str, ...]:
    """Return a tuple of all registered vector store provider names."""
    return tuple(VECTOR_STORE_REGISTRY.keys())
