"""Defines the base interface and message format for LLM chat providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TypedDict


class Message(TypedDict):
    """A single chat message with a role ('system', 'user', or 'assistant') and text content."""

    role: str
    content: str


class BaseLLM(ABC):
    """Base class that all LLM chat providers must implement."""

    @abstractmethod
    async def generate(self, messages: list[Message]) -> str:
        """Send a list of messages to the LLM and return its text response."""
        raise NotImplementedError
