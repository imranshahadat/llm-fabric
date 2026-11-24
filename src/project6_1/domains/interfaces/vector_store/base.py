"""Base interface and data structures for vector store providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, TypedDict


class VectorDocument(TypedDict):
    """document with its embedding vector and metadata."""

    id: str
    text: str
    embedding: list[float]
    metadata: dict[str, Any]


class SearchQuery(TypedDict):
    """Parameters for a vector similarity search."""

    query_embedding: list[float]
    top_k: int
    filter: dict[str, Any] | None


class SearchResult(TypedDict):
    """A single result from a vector search."""

    id: str
    score: float
    text: str
    metadata: dict[str, Any]


class BaseVectorStore(ABC):
    """Base interface that all vector store providers must implement."""

    @abstractmethod
    async def upsert(self, documents: list[VectorDocument]) -> None:
        """Insert or update documents in the vector store."""
        raise NotImplementedError

    @abstractmethod
    async def search(self, query: SearchQuery) -> list[SearchResult]:
        """Search for similar vectors using cosine similarity."""
        raise NotImplementedError

    @abstractmethod
    async def delete(self, ids: list[str]) -> None:
        """Delete documents by their IDs."""
        raise NotImplementedError
