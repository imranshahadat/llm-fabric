"""Retry helper for async LLM calls using Tenacity with exponential backoff and jitter."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TypeVar

import tenacity
from tenacity import (
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from ..exceptions.llm import LLMConnectionError, LLMError, LLMRateLimitError

T = TypeVar("T")

# Only retry errors that might succeed on a later try (rate limits and network
# drops). Do not retry auth errors or bad requests—repeating them will just fail
# again and waste quota.
_RETRYABLE: tuple[type[LLMError], ...] = (LLMRateLimitError, LLMConnectionError)


def _create_retry_decorator(
    *,
    max_attempts: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 8.0,
    retryable: tuple[type[LLMError], ...] = _RETRYABLE,
) -> tenacity.AsyncRetrying:
    """Create a Tenacity retry controller using random exponential backoff for retryable errors."""
    return tenacity.AsyncRetrying(
        stop=stop_after_attempt(max_attempts),
        wait=wait_random_exponential(
            multiplier=base_delay,
            max=max_delay,
            exp_base=2,
        ),
        retry=retry_if_exception_type(retryable),
        # if need authh error as well use this fil hal is ko commit kia hai error hai rha tha
        # ya wala error tha is ko use krny sy tests/unit/test_retry.py:59: AssertionError
        # retry=retry_if_exception_type(Exception),
        # Raise the real LLM error when retries run out instead of Tenacity's
        # generic RetryError, so upstream handlers see the actual cause.
        reraise=True,
    )


async def with_retry[T](
    fn: Callable[[], Awaitable[T]],
    *,
    max_attempts: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 8.0,
    retryable: tuple[type[LLMError], ...] = _RETRYABLE,
) -> T:
    """Run an async zero-argument function and automatically retry on transient LLM errors."""
    retryer = _create_retry_decorator(
        max_attempts=max_attempts,
        base_delay=base_delay,
        max_delay=max_delay,
        retryable=retryable,
    )

    return await retryer(fn)


# @retry(stop=stop_after_attempt(max_attempts), retry=retry_if_exception_type(retryable))
