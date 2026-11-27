"""Unit tests for the LLM-backed query summarizer."""

from __future__ import annotations

import pytest

from project6_1.core.config.settings import AppSettings
from project6_1.core.exceptions.llm import LLMRateLimitError
from project6_1.domains.infrastructure.llm.query_summarizer import LLMQuerySummarizer
from project6_1.domains.interfaces.llm.base import BaseLLM, Message
from project6_1.domains.interfaces.query_summarizer.base import BaseQuerySummarizer


def make_settings(**overrides: object) -> AppSettings:
    """Build settings with summarizer enabled by default."""
    defaults: dict[str, object] = {
        "query_summarizer_enabled": True,
        "summarizer_min_query_words": 5,
        "summarizer_max_words": 8,
    }
    defaults.update(overrides)
    return AppSettings(_env_file=None, **defaults)  # type: ignore[call-arg]


class StubLLM(BaseLLM):
    """Test double that returns a canned reply and records what it was asked."""

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls: list[list[Message]] = []

    async def generate(self, messages: list[Message]) -> str:
        self.calls.append(messages)
        return self.reply


class FailingLLM(BaseLLM):
    """Test double that always errors, simulating a small-model outage."""

    async def generate(self, messages: list[Message]) -> str:
        raise LLMRateLimitError("429 quota", provider="gemini")


LONG_QUERY = (
    "Hey so I was reading the Q3 financial report the other day and I am "
    "still wondering what exactly the total revenue figure was for the "
    "European region in that specific document"
)
SHORT_QUERY = "What was Q3 revenue?"


class TestSummarize:
    async def test_disabled_returns_original(self) -> None:
        llm = StubLLM("summary: never called")
        summarizer = LLMQuerySummarizer(make_settings(query_summarizer_enabled=False), llm)
        result = await summarizer.summarize(LONG_QUERY)
        assert result == {"text": LONG_QUERY, "was_summarized": False}
        assert llm.calls == []  # no tokens spent at all

    async def test_short_query_skips_llm(self) -> None:
        llm = StubLLM("summary: never called")
        summarizer = LLMQuerySummarizer(make_settings(), llm)
        result = await summarizer.summarize(SHORT_QUERY)
        assert result["was_summarized"] is False
        assert result["text"] == SHORT_QUERY
        assert llm.calls == []

    async def test_long_query_is_summarized(self) -> None:
        llm = StubLLM('"total revenue European region Q3"')
        summarizer = LLMQuerySummarizer(make_settings(), llm)
        result = await summarizer.summarize(LONG_QUERY)
        assert result["was_summarized"] is True
        assert result["text"] == "total revenue European region Q3"
        assert len(result["text"]) < len(LONG_QUERY)

    async def test_prompt_instructs_small_model(self) -> None:
        llm = StubLLM('"q"')
        summarizer = LLMQuerySummarizer(make_settings(), llm)
        await summarizer.summarize(LONG_QUERY)
        system = llm.calls[0][0]
        assert system["role"] == "system"
        assert "search query" in system["content"]

    async def test_prefixes_are_stripped(self) -> None:
        llm = StubLLM("Summary: the Q3 revenue figure")
        summarizer = LLMQuerySummarizer(make_settings(), llm)
        result = await summarizer.summarize(LONG_QUERY)
        assert result["text"] == "the Q3 revenue figure"

    async def test_word_cap_trims_chatty_reply(self) -> None:
        llm = StubLLM("one two three four five six seven eight nine ten")
        summarizer = LLMQuerySummarizer(make_settings(), llm)
        result = await summarizer.summarize(LONG_QUERY)
        assert len(result["text"].split()) == 8  # capped at summarizer_max_words

    async def test_llm_failure_falls_back_to_original(self) -> None:
        summarizer = LLMQuerySummarizer(make_settings(), FailingLLM())
        result = await summarizer.summarize(LONG_QUERY)
        assert result == {"text": LONG_QUERY, "was_summarized": False}

    async def test_identical_reply_means_not_summarized(self) -> None:
        # Reply within the word cap, so nothing is trimmed and text == query.
        query = "what was the total Q3 revenue"
        summarizer = LLMQuerySummarizer(make_settings(), StubLLM(query))
        result = await summarizer.summarize(query)
        assert result == {"text": query, "was_summarized": False}

    async def test_implements_port(self) -> None:
        summarizer = LLMQuerySummarizer(make_settings(), StubLLM("q"))
        assert isinstance(summarizer, BaseQuerySummarizer)

    async def test_whitespace_only_query_returned_as_is(self) -> None:
        llm = StubLLM("never called")
        summarizer = LLMQuerySummarizer(make_settings(), llm)
        result = await summarizer.summarize("   ")
        assert result == {"text": "", "was_summarized": False}


def test_settings_defaults() -> None:
    settings = AppSettings(_env_file=None)  # type: ignore[call-arg]
    assert settings.cache_enabled is True
    assert settings.cache_provider == "redis"
    assert settings.cache_ttl_seconds == 3600
    assert settings.cache_similarity_threshold == 0.92
    assert settings.query_summarizer_enabled is False
    assert settings.summarizer_min_query_words == 12
    assert settings.allowed_metadata_filter_fields == frozenset(
        {"source", "file_type", "kind", "page"}
    )


def test_settings_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CACHE_ENABLED", "false")
    monkeypatch.setenv("REDIS_URL", "rediss://default:pw@host.redis.cloud.com:6379")
    monkeypatch.setenv("CACHE_SIMILARITY_THRESHOLD", "0.85")
    monkeypatch.setenv("QUERY_SUMMARIZER_ENABLED", "true")
    monkeypatch.setenv("SUMMARIZER_MODEL", "gemini-1.5-flash-8b")
    monkeypatch.setenv("METADATA_FILTER_FIELDS", "source, kind")

    settings = AppSettings(_env_file=None)  # type: ignore[call-arg]
    assert settings.cache_enabled is False
    assert settings.redis_url.startswith("rediss://")
    assert settings.cache_similarity_threshold == 0.85
    assert settings.query_summarizer_enabled is True
    assert settings.summarizer_model == "gemini-1.5-flash-8b"
    assert settings.allowed_metadata_filter_fields == frozenset({"source", "kind"})
