"""Base interface and request/result shapes for reranker providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TypedDict


class RerankRequest(TypedDict):
    """Inputs to a rerank call: a query and the candidate documents to score."""

    query: str
    documents: list[str]


class RerankResult(TypedDict):
    """One reranked hit: the document text, its relevance score, and its new rank."""

    document: str
    score: float
    index: int


class BaseReranker(ABC):
    """Base class that all reranker providers must implement."""

    @abstractmethod
    async def rerank(
        self,
        request: RerankRequest,
        *,
        top_k: int | None = None,
    ) -> list[RerankResult]:
        """Reorder `request['documents']` by relevance to `request['query']`."""
        raise NotImplementedError
