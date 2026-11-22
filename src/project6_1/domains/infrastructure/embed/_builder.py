"""Helpers for validating and instantiating provider adapters."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, TypeVar

from ....core.config.settings import AppSettings
from ....core.exceptions.base import APIKeyMissingError

T = TypeVar("T")


def _get_nested_attribute(target_object: Any, dotted_attribute_path: str) -> Any:
    """Read a nested attribute using a dot-separated path, or None if any part is missing."""
    current_object: Any = target_object

    for attribute_name in dotted_attribute_path.split("."):
        if current_object is None:
            return None
        current_object = getattr(current_object, attribute_name, None)

    return current_object


def build_provider_adapter[T](
    *,
    provider_registry: Mapping[str, Callable[[AppSettings], T]],
    provider_name: str,
    provider_kind: str,
    api_key_setting_path: str | None,
    unknown_provider_error: type[Exception],
    application_settings: AppSettings,
) -> T:
    """Check that the provider and API key exist, then create and return the adapter."""
    if provider_name not in provider_registry:
        raise unknown_provider_error(
            f"unknown {provider_kind} provider {provider_name!r}; "
            f"known: {sorted(provider_registry)}"
        )

    if api_key_setting_path is not None:
        secret_field = _get_nested_attribute(application_settings, api_key_setting_path)
        api_key_text = secret_field.get_secret_value() if secret_field else None

        if not api_key_text:
            raise APIKeyMissingError(
                f"{provider_kind} provider {provider_name!r} requires an API key at "
                f"settings.{api_key_setting_path} — set it in .env"
            )

    adapter_factory = provider_registry[provider_name]
    return adapter_factory(application_settings)
