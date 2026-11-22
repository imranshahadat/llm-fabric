"""Google Gemini embedding model adapter built on LangChain."""

from __future__ import annotations

from ....core.config.settings import AppSettings
from ....core.exceptions.llm import LLMError
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

        # Pass the SecretStr straight through — no need to unwrap and re-wrap.
        self._client = GoogleGenerativeAIEmbeddings(
            model=self._embed_model_name,
            api_key=self._settings.gemini_api_key,
        )

    def _get_configured_model_name(self) -> str:
        """Return the Gemini embedding model name from settings, or use the default."""
        return self._settings.gemini_embedding_model or self.DEFAULT_EMBED_MODEL

    def _get_provider_api_key(self) -> str | None:
        """Return the Gemini API key as plain text, or None if unset."""
        secret_api_key = self._settings.gemini_api_key
        return secret_api_key.get_secret_value() if secret_api_key else None

    def _generate_embedding_sync(self, request: EmbedRequest) -> list[float]:
        """Request an embedding vector from Google Gemini and return it."""
        try:
            return self._client.embed_query(request["text"])
        except Exception as provider_error:
            # One line — picks LLMAuthError / LLMRateLimitError / etc. automatically.
            raise LLMError.from_provider_error(
                provider_error, provider=self.PROVIDER
            ) from provider_error
