"""Base interface for the query summarizer.

The summarizer condenses a long, chatty user question into a short,
search-focused query. Short queries save tokens twice: the small model
call is cheap, and every downstream stage (embedding, retrieval, the big
LLM) receives fewer input tokens.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TypedDict


class SummarizedQuery(TypedDict):
    """The output of a summarizer run."""

    text: str  # the condensed query to use downstream
    was_summarized: bool  # True if the text differs from the original query


class BaseQuerySummarizer(ABC):
    """Base interface that all query summarizer providers must implement."""

    @abstractmethod
    async def summarize(self, query: str) -> SummarizedQuery:
        """Condense `query` into a short search query.

        Implementations should return the original query untouched when
        summarization is disabled or the query is already short.
        """
        raise NotImplementedError
