"""Factory for creating semantic cache instances based on application settings."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

from ....core.config.settings import AppSettings, get_settings
from ....core.exceptions.base import ConfigError
from ...interfaces.semantic_cache.base import BaseSemanticCache
from .redis_cache import RedisSemanticCache

# Provider name -> adapter class.
CACHE_REGISTRY: dict[str, type[BaseSemanticCache]] = {
    "redis": RedisSemanticCache,
}


def build_semantic_cache(settings: AppSettings | None = None) -> BaseSemanticCache | None:
    """Create the configured cache, or None when caching is disabled.

    Returns None (instead of raising) when CACHE_ENABLED=false so callers
    can simply skip cache calls without a feature flag of their own.
    """
    s = settings or get_settings()
    if not s.cache_enabled:
        return None

    provider = s.cache_provider.lower()
    adapter = CACHE_REGISTRY.get(provider)
    if adapter is None:
        raise ConfigError(f"unknown cache provider {provider!r}; known: {sorted(CACHE_REGISTRY)}")
    # Cast needed because the registry values take (settings, client=None)
    # but the ABC's signature hides that from pyright.
    build = cast("Callable[[AppSettings], BaseSemanticCache]", adapter)
    return build(s)


def available_cache_providers() -> tuple[str, ...]:
    """Return a tuple of all registered cache provider names."""
    return tuple(CACHE_REGISTRY.keys())
