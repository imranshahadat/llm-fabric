"""Base class for LLM providers that handles retries, logging, and async execution."""

from __future__ import annotations

import asyncio
import time
from abc import abstractmethod

from ....core.config.settings import AppSettings
from ....core.logging.setup import get_logger
from ....core.resilience.retry import with_retry
from ...interfaces.llm.base import BaseLLM, Message


class BaseLangChainLLM(BaseLLM):
    """Base adapter that wraps provider calls with automatic retries and logging.

    Subclasses must define `PROVIDER`, `DEFAULT_MODEL`, and implement `_generate()`.
    """

    PROVIDER: str = "base"
    DEFAULT_MODEL: str = ""

    def __init__(self, settings: AppSettings) -> None:
        self._settings = settings
        self._model_name = self._resolve_model_name()
        self._log = get_logger().bind(provider=self.PROVIDER, model=self._model_name)


    @abstractmethod
    def _get_configured_model_name(self) -> str:
        """Return the configured model name, or fall back to the provider default."""
        raise NotImplementedError    

    @abstractmethod
    def _generate(self, messages: list[Message]) -> str:
        """Call the specific LLM provider and return the text response."""
        raise NotImplementedError

    def _resolve_model_name(self) -> str:
        """Return the configured model name or fall back to `DEFAULT_MODEL`."""
        return self._settings.llm_model or self.DEFAULT_MODEL

    async def generate(self, messages: list[Message]) -> str:
        """Generate a response from the LLM with automatic retries and latency logging."""
        started = time.perf_counter()
        error: Exception | None = None
        try:
            text = await with_retry(
                lambda: self._async_wrap(messages),
                max_attempts=max(1, self._settings.llm_max_retries),
            )
            return text
        except Exception as exc:
            error = exc
            raise
        finally:
            latency_ms = round((time.perf_counter() - started) * 1000, 2)
            if error is not None:
                self._log.error(
                    "llm_call",
                    latency_ms=latency_ms,
                    message_count=len(messages),
                    error=f"{type(error).__name__}: {error}",
                    error_type=type(error).__name__,
                )
            else:
                self._log.info(
                    "llm_call",
                    latency_ms=latency_ms,
                    message_count=len(messages),
                )

    async def _async_wrap(self, messages: list[Message]) -> str:
        """Run the synchronous `_generate()` method in a background thread."""
        return await asyncio.to_thread(self._generate, messages)

    def _require_key(self) -> str | None:
        """Return the API key for the specific provider, or `None` if missing."""
        raise NotImplementedError
