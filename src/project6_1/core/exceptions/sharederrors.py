"""All error types raised by LLM and embed adapters.

Each error class knows how to recognise itself from a raw provider
exception (via `matches`). Call `LLMError.from_provider_error(...)`
from any adapter and it picks the right subclass automatically.
"""

from __future__ import annotations

# Ordered list of error classes that from_provider_error() will try.
# More specific checks should come first.
_MATCHABLE: list[type[LLMError]] = []


def _matchable(cls: type[LLMError]) -> type[LLMError]:
    """Tag an error class so from_provider_error() can auto-pick it."""
    _MATCHABLE.append(cls)
    return cls


class LLMError(Exception):
    """Base error for any problem calling an LLM or embed provider."""

    def __init__(self, message: str, *, provider: str = "") -> None:
        super().__init__(message)
        self.raw_message = message
        self.provider = provider

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


@_matchable
class LLMAuthError(LLMError):
    """API key is bad or access was refused (HTTP 401/403)."""

    @staticmethod
    def matches(error: Exception) -> bool:
        text = str(error).lower()
        return any(s in text for s in ("401", "403", "permission", "api key"))


@_matchable
class LLMRateLimitError(LLMError):
    """Too many requests — rate limit or quota exhausted (HTTP 429)."""

    @staticmethod
    def matches(error: Exception) -> bool:
        text = str(error).lower()
        cls_name = type(error).__name__.lower()
        return "429" in text or "quota" in text or "rate" in cls_name


@_matchable
class LLMContextLengthError(LLMError):
    """The prompt was too long for the model's context window."""

    @staticmethod
    def matches(error: Exception) -> bool:
        text = str(error).lower()
        return "context" in text and "length" in text


@_matchable
class LLMModelUnavailableError(LLMError):
    """The model name doesn't exist on the provider (HTTP 404)."""

    @staticmethod
    def matches(error: Exception) -> bool:
        text = str(error).lower()
        return "404" in text or "not found" in text


@_matchable
class LLMInvalidRequestError(LLMError):
    """The request was malformed or had invalid params (HTTP 400)."""

    @staticmethod
    def matches(error: Exception) -> bool:
        text = str(error).lower()
        cls_name = type(error).__name__.lower()
        return "400" in text or "invalid" in cls_name


@_matchable
class LLMConnectionError(LLMError):
    """Network problem — timeout, DNS, connection refused, etc."""

    @staticmethod
    def matches(error: Exception) -> bool:
        cls_name = type(error).__name__.lower()
        return "timeout" in cls_name or "connection" in cls_name


class LLMResponseParseError(LLMError):
    """The provider replied but we couldn't parse the content.

    Not decorated with @_matchable — this is only raised explicitly
    by response-parsing code, never from a raw provider exception.
    """


class LLMProviderError(LLMError):
    """Raised when .env asks for a provider that isn't wired in the registry."""
