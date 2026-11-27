"""All error types raised by LLM and embed adapters.

Each error class knows how to recognise itself from a raw provider
exception (via `matches`). Call `LLMError.from_provider_error(...)`
from any adapter and it picks the right subclass automatically.
"""

from __future__ import annotations

from .base import AppError

# Ordered list of error classes that from_provider_error() will try.
# More specific checks should come first.
_MATCHABLE: list[type[LLMError]] = []


class LLMError(AppError):
    """Base error for any problem calling an LLM or embed provider."""

    def __init__(
        self,
        message: str,
        *,
        provider: str = "",
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.raw_message = message
        self.provider = provider
        self.status_code = status_code

    @staticmethod
    def matches(error: Exception) -> bool:
        """Return True if this class should handle the error. Base never matches."""
        return False

    @classmethod
    def from_provider_error(cls, error: Exception, *, provider: str) -> LLMError:
        """Pick the right subclass for this error and build it. Used by all adapters."""
        for error_class in _MATCHABLE:
            if error_class.matches(error):
                return error_class(str(error), provider=provider)
        # Nothing matched — wrap in a plain LLMError so callers always get one type.
        return cls(str(error), provider=provider)


class LLMAuthError(LLMError):
    """API key is bad or access was refused (HTTP 401/403)."""

    @staticmethod
    def matches(error: Exception) -> bool:
        text = str(error).lower()
        return any(s in text for s in ("401", "403", "permission", "api key"))


class LLMRateLimitError(LLMError):
    """Too many requests — rate limit or quota exhausted (HTTP 429)."""

    @staticmethod
    def matches(error: Exception) -> bool:
        text = str(error).lower()
        cls_name = type(error).__name__.lower()
        return "429" in text or "quota" in text or "rate" in cls_name


class LLMInvalidRequestError(LLMError):
    """The request was malformed or had invalid params (HTTP 400)."""

    @staticmethod
    def matches(error: Exception) -> bool:
        text = str(error).lower()
        cls_name = type(error).__name__.lower()
        return "400" in text or "invalid" in cls_name


class LLMContextLengthError(LLMInvalidRequestError):
    """The prompt was too long for the model's context window."""

    @staticmethod
    def matches(error: Exception) -> bool:
        text = str(error).lower()
        return "context" in text and "length" in text


class LLMModelUnavailableError(LLMInvalidRequestError):
    """The model name doesn't exist on the provider (HTTP 404)."""

    @staticmethod
    def matches(error: Exception) -> bool:
        text = str(error).lower()
        return "404" in text or "not found" in text


class LLMConnectionError(LLMError):
    """Network problem — timeout, DNS, connection refused, etc."""

    @staticmethod
    def matches(error: Exception) -> bool:
        cls_name = type(error).__name__.lower()
        return "timeout" in cls_name or "connection" in cls_name


class LLMResponseParseError(LLMError):
    """The provider replied but we couldn't parse the content.

    Never matched from a raw provider exception — only raised explicitly
    by response-parsing code.
    """


class LLMProviderError(LLMError):
    """Raised when .env asks for an LLM provider that isn't wired in the registry."""


# Registration order for from_provider_error(): specific classes first so a
# generic match (e.g. "invalid ...") can't steal a specific error (404, 429).
_MATCHABLE.extend(
    [
        LLMAuthError,
        LLMRateLimitError,
        LLMContextLengthError,
        LLMModelUnavailableError,
        LLMInvalidRequestError,
        LLMConnectionError,
    ]
)
