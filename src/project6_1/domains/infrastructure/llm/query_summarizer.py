"""LLM-backed query summarizer.

Uses the same chat adapters as the main LLM (any registered provider)
but with its own model + token cap, so a cheap "small" model can be
configured via SUMMARIZER_MODEL without touching the main LLM settings.
"""

from __future__ import annotations

from ....core.config.settings import AppSettings
from ....core.logging.setup import get_logger
from ...interfaces.llm.base import BaseLLM, Message
from ...interfaces.query_summarizer.base import (
    BaseQuerySummarizer,
    SummarizedQuery,
)

# Keeps the summarizer honest: strip filler the model tends to add.
_SUMMARY_PREFIXES = ("summary:", "summarized query:", "query:", "here is")


class LLMQuerySummarizer(BaseQuerySummarizer):
    """Condenses long user queries into short search queries via a small LLM."""

    def __init__(self, settings: AppSettings, llm: BaseLLM) -> None:
        self._settings = settings
        self._llm = llm
        self._log = get_logger().bind(component="query_summarizer")

    async def summarize(self, query: str) -> SummarizedQuery:
        """Return a condensed query, or the original when short or disabled.

        Errors from the small LLM never break the request: on failure the
        original query is returned with `was_summarized=False`, because a
        skipped optimization must not become an outage.
        """
        stripped = query.strip()
        if not self._settings.query_summarizer_enabled:
            return SummarizedQuery(text=stripped, was_summarized=False)

        if self._count_words(stripped) < self._settings.summarizer_min_query_words:
            # Already short — sending it to an LLM would only waste tokens.
            return SummarizedQuery(text=stripped, was_summarized=False)

        try:
            condensed = await self._llm.generate(
                [
                    Message(
                        role="system",
                        content=(
                            "Condense the user's question into one short, "
                            "self-contained search query. Keep names, numbers "
                            "and dates. Reply with the query only — no preamble."
                        ),
                    ),
                    Message(role="user", content=stripped),
                ]
            )
        except Exception as exc:
            self._log.warning(
                "summarizer_fallback_to_original",
                error=f"{type(exc).__name__}: {exc}",
            )
            return SummarizedQuery(text=stripped, was_summarized=False)

        text = self._clean(condensed)
        if not text or text.lower() == stripped.lower():
            return SummarizedQuery(text=stripped, was_summarized=False)

        saved = len(stripped) - len(text)
        self._log.info(
            "query_summarized",
            original_chars=len(stripped),
            summarized_chars=len(text),
            chars_saved=saved,
        )
        return SummarizedQuery(text=text, was_summarized=True)

    def _clean(self, text: str) -> str:
        """Strip quotes, filler prefixes and whitespace from a model reply."""
        cleaned = text.strip().strip('"').strip("'").strip()
        lowered = cleaned.lower()
        for prefix in _SUMMARY_PREFIXES:
            if lowered.startswith(prefix):
                cleaned = cleaned[len(prefix) :].lstrip(": ").strip()
                break
        # Cap the summary so a chatty model can't inflate it back.
        words = cleaned.split()
        if len(words) > self._settings.summarizer_max_words:
            cleaned = " ".join(words[: self._settings.summarizer_max_words])
        return cleaned

    @staticmethod
    def _count_words(text: str) -> int:
        """Count whitespace-separated words."""
        return len(text.split())
