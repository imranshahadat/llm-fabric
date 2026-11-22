"""Unit tests for the reranker adapter and factory.

These tests use a fake reranker so they run without any API key or network
access. A live integration test lives in scripts/verify_reranker.py.
"""

from __future__ import annotations

import asyncio

import pytest

from project6_1.core.config.settings import AppSettings, LLMProviderEnum
from project6_1.core.exceptions.llm import LLMProviderError, LLMRateLimitError
from project6_1.domains.infrastructure.reranker._base import BaseLangChainReranker
from project6_1.domains.infrastructure.reranker.factory import (
    RERANKER_REGISTRY,
    available_rerankers,
    build_reranker,
)
from project6_1.domains.interfaces.reranker.base import (
    RerankRequest,
    RerankResult,
)

# --- Fake adapters (no network calls) ------------------------------------


class _FakeReranker(BaseLangChainReranker):
    """Test double that returns documents in reversed input order."""

    PROVIDER = "fake"
    DEFAULT_RERANK_MODEL = "fake-rerank"

    def _get_configured_model_name(self) -> str:
        return self.DEFAULT_RERANK_MODEL

    def _get_provider_api_key(self) -> str | None:
        return None

    def _rerank_sync(self, request: RerankRequest, top_k: int) -> list[RerankResult]:
        reversed_docs = list(reversed(request["documents"]))
        return [
            {"document": doc, "score": 1.0 - i * 0.1, "index": i}
            for i, doc in enumerate(reversed_docs[:top_k])
        ]


class _AlwaysRateLimited(BaseLangChainReranker):
    """Test double that always raises LLMRateLimitError, to exercise retry logic."""

    PROVIDER = "fail"
    DEFAULT_RERANK_MODEL = "fail-rerank"

    def _get_configured_model_name(self) -> str:
        return self.DEFAULT_RERANK_MODEL

    def _get_provider_api_key(self) -> str | None:
        return None

    def _rerank_sync(self, request: RerankRequest, top_k: int) -> list[RerankResult]:
        raise LLMRateLimitError("429 Too Many Requests", provider=self.PROVIDER)


# --- Fixtures ------------------------------------------------------------


@pytest.fixture
def fake_settings() -> AppSettings:
    """Minimal AppSettings for tests — no real API keys, retries kept small."""
    # model_construct skips .env loading and Pydantic validation, so the test
    # is independent of whatever the developer has in their local .env.
    return AppSettings.model_construct(
        rerank_provider=LLMProviderEnum.FAKE,
        rerank_top_k=3,
        llm_max_retries=2,
        llm_timeout=1.0,
    )


@pytest.fixture
def fake_reranker(fake_settings: AppSettings) -> _FakeReranker:
    return _FakeReranker(fake_settings)


@pytest.fixture
def failing_reranker(fake_settings: AppSettings) -> _AlwaysRateLimited:
    return _AlwaysRateLimited(fake_settings)


# --- Interface contract tests -------------------------------------------


def test_rerank_request_shape() -> None:
    """RerankRequest must accept query + documents keys."""
    req: RerankRequest = {"query": "hello", "documents": ["a", "b"]}
    assert req["query"] == "hello"
    assert req["documents"] == ["a", "b"]


def test_rerank_result_shape() -> None:
    """RerankResult must accept document + score + index keys."""
    r: RerankResult = {"document": "hello", "score": 0.9, "index": 0}
    assert r["document"] == "hello"
    assert r["score"] == 0.9
    assert r["index"] == 0


# --- Base adapter behavior ----------------------------------------------


def test_rerank_returns_at_most_top_k(fake_reranker: _FakeReranker) -> None:
    """Rerank should return at most top_k results, each with the right field types."""
    request: RerankRequest = {
        "query": "irrelevant",
        "documents": ["one", "two", "three", "four", "five"],
    }
    results = asyncio.run(fake_reranker.rerank(request, top_k=3))
    assert len(results) == 3
    assert all(isinstance(r["score"], float) for r in results)
    assert all(isinstance(r["document"], str) for r in results)
    assert all(isinstance(r["index"], int) for r in results)


def test_rerank_honours_adapter_ordering(fake_reranker: _FakeReranker) -> None:
    """The fake reverses input order — verify the contract is honoured end-to-end."""
    request: RerankRequest = {
        "query": "irrelevant",
        "documents": ["a", "b", "c", "d"],
    }
    results = asyncio.run(fake_reranker.rerank(request, top_k=2))
    assert [r["document"] for r in results] == ["d", "c"]
    assert results[0]["index"] == 0  # rank 0 == most relevant


def test_rerank_uses_settings_top_k_when_none_passed(fake_reranker: _FakeReranker) -> None:
    """If top_k isn't passed, the settings default (rerank_top_k=3) is used."""
    request: RerankRequest = {
        "query": "irrelevant",
        "documents": ["a", "b", "c", "d", "e", "f"],
    }
    results = asyncio.run(fake_reranker.rerank(request))  # top_k=None -> fallback to settings
    assert len(results) == 3


def test_rerank_retries_then_raises_on_rate_limit(failing_reranker: _AlwaysRateLimited) -> None:
    """Rate-limit errors should be retried, then bubble up as LLMRateLimitError."""
    request: RerankRequest = {"query": "irrelevant", "documents": ["a", "b"]}
    with pytest.raises(LLMRateLimitError):
        asyncio.run(failing_reranker.rerank(request, top_k=2))


def test_rerank_handles_top_k_larger_than_documents(fake_reranker: _FakeReranker) -> None:
    """top_k larger than the document count should return all docs, not error."""
    request: RerankRequest = {"query": "irrelevant", "documents": ["only-one"]}
    results = asyncio.run(fake_reranker.rerank(request, top_k=10))
    assert len(results) == 1


# --- Factory tests ------------------------------------------------------


def test_cohere_is_registered_by_default() -> None:
    """Cohere must be in the default RERANKER_REGISTRY."""
    assert "cohere" in available_rerankers()
    assert "cohere" in RERANKER_REGISTRY


def test_build_reranker_raises_on_unknown_provider(fake_settings: AppSettings) -> None:
    """A provider name not in the registry must raise LLMProviderError."""
    # OPENROUTER is a valid enum value but isn't wired into RERANKER_REGISTRY.
    fake_settings.rerank_provider = LLMProviderEnum.OPENROUTER
    with pytest.raises(LLMProviderError):
        build_reranker(fake_settings)
