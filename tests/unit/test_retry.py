from __future__ import annotations

import pytest

from project6_1.core.exceptions.llm import (
    LLMAuthError,
    LLMConnectionError,
    LLMInvalidRequestError,
    LLMRateLimitError,
)
from project6_1.core.resilience.retry import with_retry


async def test_retry_succeeds_after_rate_limit() -> None:
    attempts = 0

    async def call_llm() -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise LLMRateLimitError("429", provider="openai", status_code=429)
        return "success"

    result = await with_retry(call_llm, max_attempts=3, base_delay=0.01, max_delay=0.02)
    assert result == "success"
    assert attempts == 3


async def test_retry_succeeds_after_connection_error() -> None:
    attempts = 0

    async def call_llm() -> str:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise LLMConnectionError("Connection reset", provider="gemini")
        return "recovered"

    result = await with_retry(call_llm, max_attempts=2, base_delay=0.01, max_delay=0.02)
    assert result == "recovered"
    assert attempts == 2


async def test_retry_skips_non_retryable_errors() -> None:
    for exc in (
        LLMAuthError("Unauthorized", provider="openai", status_code=401),
        LLMInvalidRequestError("Bad payload", provider="openai", status_code=400),
    ):
        attempts = 0

        async def failing_call(error: Exception = exc) -> str:
            nonlocal attempts
            attempts += 1
            raise error

        with pytest.raises(type(exc)):
            await with_retry(failing_call, max_attempts=3, base_delay=0.01, max_delay=0.02)

        assert attempts == 1


async def test_retry_exhausts_attempts_and_reraises() -> None:
    attempts = 0

    async def always_fails() -> str:
        nonlocal attempts
        attempts += 1
        raise LLMRateLimitError("Still 429", provider="gemini", status_code=429)

    with pytest.raises(LLMRateLimitError, match="Still 429"):
        await with_retry(always_fails, max_attempts=2, base_delay=0.01, max_delay=0.02)

    assert attempts == 2
