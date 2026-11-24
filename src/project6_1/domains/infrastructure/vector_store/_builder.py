"""Helper to validate API keys and instantiate provider adapters."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ....core.config.settings import AppSettings
from ....core.exceptions.base import APIKeyMissingError


def _resolve_attr(obj: Any, dotted_path: str) -> Any:
    """Safely access a nested attribute using a dot-separated path."""
    current: Any = obj
    for part in dotted_path.split("."):
        if current is None:
            return None
        current = getattr(current, part, None)
    return current


def build_provider_adapter[T](
    *,
    provider_registry: dict[str, Callable[[AppSettings], T]],
    provider_name: str,
    provider_kind: str,
    api_key_setting_path: str | None,
    application_settings: AppSettings,
    unknown_provider_error: type[Exception] = RuntimeError,
) -> T:
    """Validate API key and instantiate a provider adapter."""
    if provider_name not in provider_registry:
        known_providers = sorted(provider_registry.keys())
        raise unknown_provider_error(
            f"Unknown {provider_kind} provider {provider_name!r}; "
            f"known providers: {known_providers}"
        )

    if api_key_setting_path is not None:
        secret = _resolve_attr(application_settings, api_key_setting_path)
        api_key_value = secret.get_secret_value() if secret else None

        if not api_key_value:
            raise APIKeyMissingError(
                f"{provider_kind} provider {provider_name!r} requires an API key "
                f"at settings.{api_key_setting_path} — set it in .env"
            )

    adapter_cls = provider_registry[provider_name]
    return adapter_cls(application_settings)
