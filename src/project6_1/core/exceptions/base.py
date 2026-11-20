"""Base application exception hierarchy."""

from __future__ import annotations


class AppError(Exception):
    """Base error for all intentional application failures."""


class ConfigError(AppError):
    """Configuration values or environment variables are missing or invalid."""


class APIKeyMissingError(ConfigError):
    """A required API key is missing or empty."""


class ProviderError(AppError):
    """A provider could not be loaded, configured, or initialized."""
