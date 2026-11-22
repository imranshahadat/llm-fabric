"""Helper to look up, validate API keys for, and instantiate provider adapters."""

from __future__ import annotations

from typing import Any, TypeVar

from ....core.exceptions.base import APIKeyMissingError

T = TypeVar("T")


def _resolve_attr(obj: Any, dotted: str) -> Any:
    """Safely access a nested attribute using a dot-separated path (e.g., 'vercel.api_key')."""
    cur: Any = obj
    for part in dotted.split("."):
        if cur is None:
            return None
        cur = getattr(cur, part, None)
    return cur


def build_provider[T](
    *,
    registry: dict[str, type[T]],
    provider: str,
    kind: str,
    api_key_path: str | None,
    error: type[Exception],
    settings: Any,
) -> T:
    """Find a provider class in the registry, verify its API key if needed, and return an instance."""
    if provider not in registry:
        raise error(f"unknown {kind} provider {provider!r}; known: {sorted(registry)}")
    if api_key_path is not None:
        secret = _resolve_attr(settings, api_key_path)
        value = secret.get_secret_value() if secret else None
        if not value:
            raise APIKeyMissingError(
                f"{kind} provider {provider!r} requires an API key at "
                f"settings.{api_key_path} — set it in .env"
            )
    adapter_cls = registry[provider]
    return adapter_cls(settings)  # type: ignore[call-arg]
