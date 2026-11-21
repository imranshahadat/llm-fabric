"""Google Gemini embedding model adapter built on LangChain."""

from __future__ import annotations

from pydantic import SecretStr

from ....core.config.settings import AppSettings
from ....core.exceptions.llm import (
    LLMAuthError,
    LLMConnectionError,
    LLMContextLengthError,
    LLMError,
    LLMInvalidRequestError,
    LLMModelUnavailableError,
    LLMRateLimitError,
)
from ...interfaces.embed.base import EmbedRequest
from ._base import BaseLangChainEmbed


class GeminiEmbeddingModel(BaseLangChainEmbed):
    """Adapter that generates text embeddings using Google Gemini via LangChain."""

    PROVIDER = "gemini"
    DEFAULT_EMBED_MODEL = "gemini-embedding-001"

    def __init__(self, settings: AppSettings) -> None:
        super().__init__(settings)

        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
        except ImportError as import_error:  # pragma: no cover
            raise LLMError(
                "langchain-google-genai is not installed; run `uv sync --extra gemini`",
                provider=self.PROVIDER,
            ) from import_error

        api_key_value = self._get_provider_api_key()
        self._client = GoogleGenerativeAIEmbeddings(
            model=self._embed_model_name,
            api_key=SecretStr(api_key_value) if api_key_value else None,
        )

    def _get_configured_model_name(self) -> str:
        """Return the configured Gemini embedding model name, or fall back to the default."""
        return self._settings.gemini_embedding_model or self.DEFAULT_EMBED_MODEL

    def _get_provider_api_key(self) -> str | None:
        """Return the configured Gemini API key as plain text, or None if unset."""
        secret_api_key = self._settings.gemini_api_key
        return secret_api_key.get_secret_value() if secret_api_key else None

    def _generate_embedding_sync(self, request: EmbedRequest) -> list[float]:
        """Synchronously request an embedding vector from Google Gemini."""
        try:
            return self._client.embed_query(request["text"])
        except Exception as provider_error:
            raise _convert_provider_exception(
                provider_error,
                provider_name=self.PROVIDER,
            ) from provider_error


def _convert_provider_exception(
    provider_error: Exception,
    *,
    provider_name: str,
) -> LLMError:
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
