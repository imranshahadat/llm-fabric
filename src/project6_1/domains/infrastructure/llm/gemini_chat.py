"""Google Gemini LLM adapter using LangChain."""

from __future__ import annotations

from typing import Any

from ....core.exceptions.llm import (
    LLMAuthError,
    LLMConnectionError,
    LLMContextLengthError,
    LLMError,
    LLMInvalidRequestError,
    LLMModelUnavailableError,
    LLMRateLimitError,
    LLMResponseParseError,
)
from ...interfaces.llm.base import Message
from ._base import BaseLangChainLLM


class GeminiLLM(BaseLangChainLLM):
    """Handles chat completions for Google Gemini models."""

    PROVIDER = "gemini"
    DEFAULT_MODEL = "gemini-1.5-flash"

    def _resolve_model_name(self) -> str:
        return self._settings.gemini_model or self.DEFAULT_MODEL

    def _generate(self, messages: list[Message]) -> str:
        key = self._settings.gemini_api_key
        api_key = key.get_secret_value() if key else None
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as exc:  # pragma: no cover
            raise LLMError(
                "langchain-google-genai is not installed; run `uv sync --extra gemini`",
                provider=self.PROVIDER,
            ) from exc

        lc_messages = [_to_lc(m) for m in messages]
        try:
            client = ChatGoogleGenerativeAI(
                model=self._model_name,
                google_api_key=api_key,
                temperature=self._settings.llm_temperature,
                max_output_tokens=self._settings.llm_max_tokens,
                timeout=self._settings.llm_timeout,
            )
            result = client.invoke(lc_messages)
        except Exception as exc:
            raise _map_gemini_error(exc, self.PROVIDER) from exc

        # `invoke()` is statically typed to return an `AIMessage`, so an
        # `isinstance(result, AIMessage)` check would be flagged by the type
        # checker as redundant. The real failure mode at runtime is an
        # unexpected *content* shape (`content` is `str | list[...]`), which
        # `_content_to_str` defends against below.
        return _content_to_str(result.content, provider=self.PROVIDER)


def _to_lc(m: Message):
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    role = m["role"]
    content = m["content"]
    if role == "system":
        return SystemMessage(content=content)
    if role == "user":
        return HumanMessage(content=content)
    if role == "assistant":
        return AIMessage(content=content)
    raise LLMInvalidRequestError(f"unknown role {role!r}")


def _content_to_str(
    content: str | list[str | dict[Any, Any]],
    *,
    provider: str,
) -> str:
    """Convert a string or list of content blocks from the model response into plain text."""
    if isinstance(content, str):
        return content

    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
            continue
        # Per the ``str | dict[Any, Any]`` element union, ``block`` is a
        # dict here, so no further isinstance is needed — pyright narrows it
        # automatically after the ``str`` branch above.
        if block.get("type") == "text" and isinstance(block.get("text"), str):
            parts.append(block["text"])
        else:
            parts.append(str(block))

    text = "".join(parts)
    if not text:
        raise LLMResponseParseError(
            "empty response content",
            provider=provider,
        )
    return text


def _map_gemini_error(exc: Exception, provider: str) -> LLMError:
    name = type(exc).__name__
    msg = str(exc).lower()
    if "401" in msg or "403" in msg or "permission" in msg or "api key" in msg:
        return LLMAuthError(str(exc), provider=provider)
    if "429" in msg or "quota" in msg or "rate" in name.lower():
        return LLMRateLimitError(str(exc), provider=provider)
    if "context" in msg and "length" in msg:
        return LLMContextLengthError(str(exc), provider=provider)
    if "404" in msg or "not found" in msg:
        return LLMModelUnavailableError(str(exc), provider=provider)
    if "400" in msg or "invalid" in name.lower():
        return LLMInvalidRequestError(str(exc), provider=provider)
    if "timeout" in name.lower() or "connection" in name.lower():
        return LLMConnectionError(str(exc), provider=provider)
    return LLMError(str(exc), provider=provider)
