"""Base interface for embedding models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TypedDict


class EmbedRequest(TypedDict):
    """Input payload containing the text to turn into an embedding vector."""

    text: str


class BaseEmbed(ABC):
    """Base interface for all embedding models."""

    @abstractmethod
    async def generate_embedding(self, request: EmbedRequest) -> list[float]:
        """Generate an embedding vector from the input text."""
        raise NotImplementedError