"""Base embedding adapter for LangChain-compatible providers.

Provides shared retry logic, latency logging, and non-blocking execution.
Subclasses implement `_generate_embedding_sync()` for provider-specific calls.
"""

from __future__ import annotations

import asyncio
import time
from abc import abstractmethod

from ....core.logging.setup import get_logger
from ....core.config.settings import AppSettings
from ....core.resilience.retry import with_retry
from ...interfaces.embed.base import BaseEmbed, EmbedRequest


class BaseLangChainEmbed(BaseEmbed):
    """Shared base class for LangChain embedding providers."""

    PROVIDER: str = "base"
    DEFAULT_EMBED_MODEL: str = ""

    def __init__(self, settings: AppSettings) -> None:
        self._settings = settings
        self._embed_model_name = self._get_configured_model_name()

        self._log = get_logger().bind(
            provider=self.PROVIDER,
            model=self._embed_model_name,
        )

    @abstractmethod
    def _generate_embedding_sync(self, request: EmbedRequest) -> list[float]:
        """Synchronously call the provider SDK to produce an embedding vector."""
        raise NotImplementedError

    async def generate_embedding(self, request: EmbedRequest) -> list[float]:
        """Generate an embedding vector with automatic retries and latency logging."""
        started = time.perf_counter()
        error: Exception | None = None

        # Create an async wrapper that properly awaits the thread call
        async def _call_with_retry():
            return await asyncio.to_thread(self._generate_embedding_sync, request)

        try:
            return await with_retry(_call_with_retry, max_attempts=max(1, self._settings.llm_max_retries))
        except Exception as exc:
            error = exc
            raise
        finally:
            latency_ms = round((time.perf_counter() - started) * 1000, 2)

            if error is not None:
                self._log.error(
                    "embedding_call",
                    latency_ms=latency_ms,
                    error=f"{type(error).__name__}: {error}",
                    error_type=type(error).__name__,
                )
            else:
                self._log.info(
                    "embedding_call",
                    latency_ms=latency_ms,
                )

    @abstractmethod
    def _get_configured_model_name(self) -> str:
        """Return the embedding model name from settings or the provider default."""
        raise NotImplementedError

    @abstractmethod
    def _get_provider_api_key(self) -> str | None:
        """Return the provider API key from settings, or None if not needed."""
        raise NotImplementedError