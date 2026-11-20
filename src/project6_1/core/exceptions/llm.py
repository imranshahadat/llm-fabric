"""Runtime and provider-level exceptions for LLM calls."""

from __future__ import annotations

from .base import AppError, ProviderError


class LLMProviderError(ProviderError):
    """LLM factory or provider resolution failed (for example, an unknown provider)."""

    def __init__(self, message: str, *, provider: str | None = None) -> None:
        super().__init__(message)
        self.provider = provider


class LLMError(AppError):
    """Base error for failures that happen while executing an LLM request."""

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code


class LLMRateLimitError(LLMError):
    """Provider rate limit hit (HTTP 429). Safe to retry after backoff."""


class LLMConnectionError(LLMError):
    """Network failure, timeout, or server error (HTTP 5xx). Safe to retry."""


class LLMAuthError(LLMError):
    """Authentication or permission failed (HTTP 401/403). Fix credentials first."""


class LLMInvalidRequestError(LLMError):
    """Request rejected due to bad parameters or input (HTTP 400). Do not retry as-is."""


class LLMContextLengthError(LLMInvalidRequestError):
    """Prompt and conversation history exceed the model's context window."""


class LLMModelUnavailableError(LLMInvalidRequestError):
    """Target model does not exist, was retired, or is unavailable (HTTP 404)."""


class LLMResponseParseError(LLMError):
    """Provider returned a response that could not be decoded or parsed."""
