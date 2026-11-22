"""OpenRouter embedding adapter — OpenAI-compatible gateway built on LangChain."""

from __future__ import annotations

from ....core.config.settings import AppSettings
from ....core.exceptions.llm import LLMError
from ...interfaces.embed.base import EmbedRequest
from ._base import BaseLangChainEmbed


class OpenRouterEmbeddingModel(BaseLangChainEmbed):
    """Adapter that generates text embeddings via OpenRouter."""

    PROVIDER = "openrouter"
    DEFAULT_EMBED_MODEL = "openai/text-embedding-3-small"

    def __init__(self, settings: AppSettings) -> None:
        super().__init__(settings)

        try:
            from langchain_openai import OpenAIEmbeddings
        except ImportError as import_error:  # pragma: no cover
            raise LLMError(
                "langchain-openai is not installed; run `uv sync --extra openai`",
                provider=self.PROVIDER,
            ) from import_error

        # OpenRouter's two ranking headers (optional but recommended).
        headers = {
            "HTTP-Referer": self._settings.openrouter_site_url,
            "X-Title": self._settings.openrouter_app_name,
        }

        # Pass the SecretStr straight through — OpenAIEmbeddings.api_key wants SecretStr.
        self._client = OpenAIEmbeddings(
            model=self._embed_model_name,
            api_key=self._settings.openrouter_api_key,
            base_url=self._settings.openrouter_base_url,
            default_headers=headers,
        )

    def _get_configured_model_name(self) -> str:
        """Return the OpenRouter embedding model name from settings, or use the default."""
        return self._settings.openrouter_embedding_model or self.DEFAULT_EMBED_MODEL

    def _get_provider_api_key(self) -> str | None:
        """Return the OpenRouter API key as plain text, or None if unset."""
        secret_api_key = self._settings.openrouter_api_key
        return secret_api_key.get_secret_value() if secret_api_key else None

    def _generate_embedding_sync(self, request: EmbedRequest) -> list[float]:
        """Request an embedding vector from OpenRouter and return it."""
        try:
            return self._client.embed_query(request["text"])
        except Exception as provider_error:
            # One line — picks LLMAuthError / LLMRateLimitError / etc. automatically.
            raise LLMError.from_provider_error(
                provider_error, provider=self.PROVIDER
            ) from provider_error
