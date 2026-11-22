from __future__ import annotations

import pytest

from project6_1.core.exceptions.base import (
    APIKeyMissingError,
    AppError,
    ConfigError,
    ProviderError,
)
from project6_1.core.exceptions.llm import (
    LLMAuthError,
    LLMConnectionError,
    LLMContextLengthError,
    LLMError,
    LLMInvalidRequestError,
    LLMModelUnavailableError,
    LLMProviderError,
    LLMRateLimitError,
    LLMResponseParseError,
)


def test_base_exception_hierarchy() -> None:
    assert issubclass(ConfigError, AppError)
    assert issubclass(APIKeyMissingError, ConfigError)
    assert issubclass(ProviderError, AppError)


def test_llm_exception_hierarchy() -> None:
    assert issubclass(LLMProviderError, LLMError)

    assert issubclass(LLMError, AppError)
    assert issubclass(LLMRateLimitError, LLMError)
    assert issubclass(LLMConnectionError, LLMError)
    assert issubclass(LLMAuthError, LLMError)
    assert issubclass(LLMInvalidRequestError, LLMError)
    assert issubclass(LLMContextLengthError, LLMInvalidRequestError)
    assert issubclass(LLMModelUnavailableError, LLMInvalidRequestError)
    assert issubclass(LLMResponseParseError, LLMError)


def test_llm_error_attributes_and_catching() -> None:
    err = LLMRateLimitError("rate limited", provider="gemini")
    assert err.provider == "gemini"
    assert "429" in str(err) or "rate" in str(err).lower()

    with pytest.raises(LLMError):
        raise err

    with pytest.raises(AppError):
        raise ConfigError("Bad config")
