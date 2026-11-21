"""Google Gemini chat model adapter built on LangChain."""

from __future__ import annotations

from typing import Any, TypedDict

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


class ChatMessage(TypedDict):
    """Single chat turn passed to the language model."""

    role: str
    content: str


class GeminiChatModel(BaseLangChainLLM):
    """Adapter that sends chat messages to Google Gemini via LangChain."""

    PROVIDER = "gemini"
    DEFAULT_MODEL = "gemini-1.5-flash"

    def _get_configured_model_name(self) -> str:
        """Return the configured Gemini model name, or fall back to the default."""
        return self._settings.gemini_model or self.DEFAULT_MODEL

    def _generate(self, messages: list[Message]) -> str:
        """Send chat messages to Gemini synchronously and return the text response."""
        secret_api_key = self._settings.gemini_api_key
        api_key_value = secret_api_key.get_secret_value() if secret_api_key else None

        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as import_error:
            raise LLMError(
                "langchain-google-genai is not installed; run `uv sync --extra gemini`",
                provider=self.PROVIDER,
            ) from import_error

        # Convert Message objects to ChatMessage dicts
        chat_messages: list[ChatMessage] = [
            {"role": msg["role"], "content": msg["content"]} for msg in messages
        ]
        langchain_messages = [_convert_message_to_langchain(msg) for msg in chat_messages]

        try:
            gemini_client = ChatGoogleGenerativeAI(
                model=self._model_name,
                google_api_key=api_key_value,
                temperature=self._settings.llm_temperature,
                max_output_tokens=self._settings.llm_max_tokens,
                timeout=self._settings.llm_timeout,
            )
            response = gemini_client.invoke(langchain_messages)
        except Exception as provider_error:
            raise _convert_provider_exception(
                provider_error,
                provider_name=self.PROVIDER,
            ) from provider_error

        return _extract_text_from_content(
            response.content,
            provider_name=self.PROVIDER,
        )


def _convert_message_to_langchain(message: ChatMessage) -> Any:
    """Convert an application chat message into a LangChain message object."""
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    message_role = message["role"]
    message_content = message["content"]

    if message_role == "system":
        return SystemMessage(content=message_content)
    if message_role == "user":
        return HumanMessage(content=message_content)
    if message_role == "assistant":
        return AIMessage(content=message_content)

    raise LLMInvalidRequestError(f"unknown message role {message_role!r}")


def _extract_text_from_content(
    response_content: str | list[str | dict[Any, Any]],
    *,
    provider_name: str,
) -> str:
    """Extract plain text from a string or content-block list returned by the model."""
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


def _convert_provider_exception(provider_error: Exception, *, provider_name: str) -> LLMError:
    """Map a raw SDK or network exception into a standardized application LLMError."""
    exception_class_name = type(provider_error).__name__
    error_message_lower = str(provider_error).lower()
    raw_error_message = str(provider_error)

    if (
        "401" in error_message_lower
        or "403" in error_message_lower
        or "permission" in error_message_lower
        or "api key" in error_message_lower
    ):
        return LLMAuthError(raw_error_message, provider=provider_name)

    if (
        "429" in error_message_lower
        or "quota" in error_message_lower
        or "rate" in exception_class_name.lower()
    ):
        return LLMRateLimitError(raw_error_message, provider=provider_name)

    if "context" in error_message_lower and "length" in error_message_lower:
        return LLMContextLengthError(raw_error_message, provider=provider_name)

    if "404" in error_message_lower or "not found" in error_message_lower:
        return LLMModelUnavailableError(raw_error_message, provider=provider_name)

    if "400" in error_message_lower or "invalid" in exception_class_name.lower():
        return LLMInvalidRequestError(raw_error_message, provider=provider_name)

    if "timeout" in exception_class_name.lower() or "connection" in exception_class_name.lower():
        return LLMConnectionError(raw_error_message, provider=provider_name)

    return LLMError(raw_error_message, provider=provider_name)
