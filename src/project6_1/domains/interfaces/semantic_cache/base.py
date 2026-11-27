"""Base interface for the semantic answer cache.

A semantic cache stores LLM answers keyed by the query's embedding
vector. On a new query it looks up nearby stored queries — if one is
similar enough (cosine similarity ≥ threshold), the cached answer is
returned without calling the LLM, saving both tokens and latency.

Entries live in Redis (Redis Cloud works the same as self-hosted) under
a namespace, with a per-entry TTL, and can be scoped (e.g. per tenant or
per document set) so answers never leak across scopes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TypedDict


class CacheHit(TypedDict):
    """A successful cache lookup."""

    answer: str  # the cached LLM answer
    similarity: float  # cosine similarity between the two query embeddings
    cache_key: str  # Redis key of the matched entry


class CacheEntry(TypedDict):
    """One cached query/answer pair, as stored in Redis."""

    query: str
    answer: str
    embedding: list[float]


class BaseSemanticCache(ABC):
    """Base interface that all semantic cache providers must implement."""

    @abstractmethod
    async def lookup(self, embedding: list[float], *, scope: str = "default") -> CacheHit | None:
        """Return the closest cached answer within the threshold, else None."""
        raise NotImplementedError

    @abstractmethod
    async def store(
        self,
        *,
        query: str,
        answer: str,
        embedding: list[float],
        scope: str = "default",
    ) -> None:
        """Cache an answer for later similarity lookup."""
        raise NotImplementedError

    @abstractmethod
    async def clear(self, scope: str = "default") -> int:
        """Delete all cached entries in a scope. Return how many were removed."""
        raise NotImplementedError

    @abstractmethod
    async def ping(self) -> bool:
        """Return True if the cache backend is reachable."""
        raise NotImplementedError
