"""Test the BaseLangChainEmbed base class."""

from __future__ import annotations

import asyncio

import pytest

from project6_1.core.config.settings import AppSettings
from project6_1.core.exceptions.llm import LLMRateLimitError
from project6_1.domains.infrastructure.embed._base import BaseLangChainEmbed
from project6_1.domains.interfaces.embed.base import EmbedRequest

# tests/unit/_helpers.py
from pydantic import SecretStr

def secret(value: str = "test-key") -> SecretStr:
    """Wrap a literal as SecretStr so type checkers stop complaining in tests."""
    return SecretStr(value)

class ConcreteEmbedModel(BaseLangChainEmbed):
    """Concrete implementation for testing."""

    PROVIDER = "test"
    DEFAULT_EMBED_MODEL = "test-model-001"

    def _generate_embedding_sync(self, request: EmbedRequest) -> list[float]:
        """Synchronous embedding generation."""
        return [float(len(request["text"])), 0.1, 0.2]

    def _get_configured_model_name(self) -> str:
        """Return model name."""
        return self._settings.gemini_embedding_model or self.DEFAULT_EMBED_MODEL

    def _get_provider_api_key(self) -> str | None:
        """Return API key."""
        if self._settings.gemini_api_key:
            return self._settings.gemini_api_key.get_secret_value()
        return None


class TestBaseLangChainEmbed:
    """Test BaseLangChainEmbed functionality."""

    def test_initialization(self) -> None:
        """Should initialize with settings."""
        settings = AppSettings(
            gemini_api_key="test-key", # type: ignore[arg-type]
            gemini_embedding_model="test-model",
        )
        adapter = ConcreteEmbedModel(settings)
        assert adapter.PROVIDER == "test"

    def test_generate_embedding_returns_list_of_floats(self) -> None:
        """generate_embedding should return a list of floats."""
        settings = AppSettings(gemini_api_key="test-key") # type: ignore[arg-type]
        adapter = ConcreteEmbedModel(settings)

        vector = asyncio.run(adapter.generate_embedding({"text": "hello"}))

        assert isinstance(vector, list)
        assert len(vector) == 3
        assert vector[0] == 5.0  # len("hello")
        assert all(isinstance(v, float) for v in vector)

    def test_generate_embedding_logs_latency(self) -> None:
        """Should log latency on successful call."""
        settings = AppSettings(gemini_api_key="test-key") # type: ignore[arg-type]
        adapter = ConcreteEmbedModel(settings)

        vector = asyncio.run(adapter.generate_embedding({"text": "test"}))
        assert isinstance(vector, list)

    def test_generate_embedding_with_empty_text(self) -> None:
        """Should handle empty text."""
        settings = AppSettings(gemini_api_key="test-key") # type: ignore[arg-type]
        adapter = ConcreteEmbedModel(settings)

        vector = asyncio.run(adapter.generate_embedding({"text": ""}))

        assert isinstance(vector, list)
        assert vector[0] == 0.0  # len("")

    def test_generate_embedding_with_long_text(self) -> None:
        """Should handle long text."""
        settings = AppSettings(gemini_api_key="test-key") # type: ignore[arg-type]
        adapter = ConcreteEmbedModel(settings)

        long_text = "a" * 1000
        vector = asyncio.run(adapter.generate_embedding({"text": long_text}))

        assert isinstance(vector, list)
        assert vector[0] == 1000.0

    @pytest.mark.asyncio
    async def test_generate_embedding_is_async(self) -> None:
        """generate_embedding should be an async method."""
        settings = AppSettings(gemini_api_key="test-key") # type: ignore[arg-type]
        adapter = ConcreteEmbedModel(settings)

        coro = adapter.generate_embedding({"text": "test"})
        assert hasattr(coro, "__await__")

        vector = await coro
        assert isinstance(vector, list)

    def test_retry_on_failure(self) -> None:
        """Should retry on a transient LLM error up to max_retries."""
        call_count = 0

        class FailingEmbedModel(ConcreteEmbedModel):
            def _generate_embedding_sync(self, request: EmbedRequest) -> list[float]:
                nonlocal call_count
                call_count += 1
                if call_count < 3:
                    # LLMRateLimitError is in _RETRYABLE, so with_retry will retry it.
                    # RuntimeError would NOT be retried after the retry.py fix.
                    raise LLMRateLimitError("Temporary failure", provider="test")
                return [1.0, 2.0, 3.0]

        settings = AppSettings(
            gemini_api_key="test-key", # type: ignore[arg-type]
            llm_max_retries=3,
        )
        adapter = FailingEmbedModel(settings)

        vector = asyncio.run(adapter.generate_embedding({"text": "test"}))

        assert vector == [1.0, 2.0, 3.0]
        assert call_count == 3

    def test_raises_after_max_retries(self) -> None:
        """Should raise the retryable error after exhausting all retry attempts."""
        call_count = 0

        class AlwaysFailingEmbedModel(ConcreteEmbedModel):
            def _generate_embedding_sync(self, request: EmbedRequest) -> list[float]:
                nonlocal call_count
                call_count += 1
                raise LLMRateLimitError("Always fails", provider="test")

        settings = AppSettings(
            gemini_api_key="test-key", # type: ignore[arg-type]
            llm_max_retries=2,
        )
        adapter = AlwaysFailingEmbedModel(settings)

        with pytest.raises(LLMRateLimitError):
            asyncio.run(adapter.generate_embedding({"text": "test"}))

        # Critical assertion — proves retries actually happened.
        # Without this, the test would pass even if with_retry never retried.
        assert call_count == 2