"""Base reranker adapter — shared retry logic, latency logging, and async wrapping."""

from __future__ import annotations

import asyncio
import time
from abc import abstractmethod

from ....core.config.settings import AppSettings
from ....core.exceptions.llm import LLMError
from ....core.logging.setup import get_logger
from ....core.resilience.retry import with_retry
from ...interfaces.reranker.base import BaseReranker, RerankRequest, RerankResult


class BaseLangChainReranker(BaseReranker):
    """Shared base class for reranker providers."""

    PROVIDER: str = "base"
    DEFAULT_RERANK_MODEL: str = ""

    def __init__(self, settings: AppSettings) -> None:
        self._settings = settings
        self._rerank_model_name = self._get_configured_model_name()
        self._log = get_logger().bind(
            provider=self.PROVIDER,
            model=self._rerank_model_name,
        )

    @abstractmethod
    def _get_configured_model_name(self) -> str:
        """Return the rerank model name from settings, or the provider default."""
        raise NotImplementedError

    @abstractmethod
    def _get_provider_api_key(self) -> str | None:
        """Return the provider API key from settings, or None if not needed."""
        raise NotImplementedError

    @abstractmethod
    def _rerank_sync(self, request: RerankRequest, top_k: int) -> list[RerankResult]:
        """Synchronously call the provider SDK to rerank documents."""
        raise NotImplementedError

    async def rerank(
        self,
        request: RerankRequest,
        *,
        top_k: int | None = None,
    ) -> list[RerankResult]:
        """Rerank documents by relevance to the query, with retries and latency logging."""
        started = time.perf_counter()
        error: Exception | None = None
        effective_top_k = top_k or self._settings.rerank_top_k

        async def _call_with_retry() -> list[RerankResult]:
            """Run the provider call in a thread; map any error to our typed LLMError."""
            try:
                return await asyncio.to_thread(self._rerank_sync, request, effective_top_k)
            except Exception as raw_err:
                raise LLMError.from_provider_error(raw_err, provider=self.PROVIDER) from raw_err

        try:
            return await with_retry(
                _call_with_retry, max_attempts=max(1, self._settings.llm_max_retries)
            )
        except Exception as exc:
            error = exc
            raise
        finally:
            latency_ms = round((time.perf_counter() - started) * 1000, 2)
            if error is not None:
                self._log.error(
                    "rerank_call",
                    latency_ms=latency_ms,
                    doc_count=len(request["documents"]),
                    error=f"{type(error).__name__}: {error}",
                    error_type=type(error).__name__,
                )
            else:
                self._log.info(
                    "rerank_call",
                    latency_ms=latency_ms,
                    doc_count=len(request["documents"]),
                )
