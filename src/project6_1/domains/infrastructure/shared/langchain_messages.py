"""Shared helpers for LangChain-based LLM adapters.

Every chat adapter (Gemini, OpenRouter, Vercel, etc.) needs to do two things:
1. Turn our app's Message into a LangChain message before calling the model.
2. Pull plain text out of whatever LangChain gives back.

These two helpers do exactly that, so each adapter only has to write the
provider-specific client call itself.
"""

from __future__ import annotations

from typing import Any

from ....core.exceptions.llm import LLMInvalidRequestError, LLMResponseParseError
from ...interfaces.llm.base import Message


def convert_message_to_langchain(message: Message) -> Any:
    """Turn one app message into the LangChain message object the model expects."""
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    role = message["role"]
    content = message["content"]

    if role == "system":
        return SystemMessage(content=content)
    if role == "user":
        return HumanMessage(content=content)
    if role == "assistant":
        return AIMessage(content=content)

    raise LLMInvalidRequestError(f"unknown message role {role!r}")


def extract_text_from_content(
    response_content: str | list[str | dict[Any, Any]],
    *,
    provider_name: str,
) -> str:
    """Pull plain text out of whatever the model returned (a string or a list of blocks)."""
    if isinstance(response_content, str):
        return response_content

    text_parts: list[str] = []
    for content_block in response_content:
        if isinstance(content_block, str):
            text_parts.append(content_block)
            continue

        if content_block.get("type") == "text" and isinstance(content_block.get("text"), str):
            text_parts.append(content_block["text"])
        else:
            text_parts.append(str(content_block))

    combined_text = "".join(text_parts)
    if not combined_text:
        raise LLMResponseParseError(
            "empty response content",
            provider=provider_name,
        )
    return combined_text
