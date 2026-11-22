"""Factory that builds the configured reranker adapter."""

from __future__ import annotations

from ....core.config.settings import AppSettings, get_settings
from ....core.exceptions.llm import LLMProviderError
from ...interfaces.reranker.base import BaseReranker
from ._builder import build_provider
from .cohere_rerank import CohereReranker

# Provider name -> adapter class.
# Keys MUST cover every value of LLMProviderEnum that you actually ship.
# Not yet wired: gemini, openrouter, vercel — add when you implement them.
RERANKER_REGISTRY: dict[str, type[BaseReranker]] = {
    "cohere": CohereReranker,
}

# Provider name -> dotted path into AppSettings for the required API key.
API_KEY_REQUIREMENTS: dict[str, str] = {
    "cohere": "cohere_api_key",
}


def build_reranker(settings: AppSettings | None = None) -> BaseReranker:
    """Build the reranker adapter chosen by RERANKER_PROVIDER in .env."""
    s = settings or get_settings()
    provider = s.rerank_provider.value.lower()
    return build_provider(
        registry=RERANKER_REGISTRY,
        provider=provider,
        kind="Reranker",
        api_key_path=API_KEY_REQUIREMENTS.get(provider),
        error=LLMProviderError,
        settings=s,
    )


def available_rerankers() -> tuple[str, ...]:
    """Return the names of all rerankers currently wired in RERANKER_REGISTRY."""
    return tuple(RERANKER_REGISTRY.keys())
