"""Vercel chat adapter — OpenAI-compatible gateway built on LangChain."""

from __future__ import annotations

from ....core.exceptions.llm import LLMError
from ...interfaces.llm.base import Message
from ..shared.langchain_messages import (
    convert_message_to_langchain,
    extract_text_from_content,
)
from ._base import BaseLangChainLLM


class VercelRouterLLM(BaseLangChainLLM):
    """Adapter that routes chat calls through Vercel."""

    PROVIDER = "vercel"
    DEFAULT_MODEL = "openai/gpt-4o-mini"

    def _get_configured_model_name(self) -> str:
        """Return the Vercel model name from settings, or use the default."""
        return self._settings.vercel_model or self.DEFAULT_MODEL

    def _generate(self, messages: list[Message]) -> str:
        """Send chat messages to Vercel and return the text reply."""
        # ChatOpenAI.api_key is typed as SecretStr | None — pass it directly,
        # do NOT unwrap to str (that's only needed for the Gemini SDK).
        secret = self._settings.vercel_api_key

        try:
            from langchain_openai import ChatOpenAI
        except ImportError as err:  # pragma: no cover
            raise LLMError(
                "langchain-openai is not installed; run `uv sync --extra openai`",
                provider=self.PROVIDER,
            ) from err

        langchain_messages = [convert_message_to_langchain(msg) for msg in messages]

        # Vercel's two ranking headers (optional but recommended).
        headers = {
            "HTTP-Referer": self._settings.vercel_base_url,
            "X-Title": self._settings.vercel_model,
        }

        # Pass max_tokens via model_kwargs — Pylance's stub for ChatOpenAI
        # doesn't expose max_tokens as a direct param in strict mode.
        model_kwargs: dict[str, int] = {}
        if self._settings.llm_max_tokens is not None:
            model_kwargs["max_tokens"] = self._settings.llm_max_tokens

        try:
            client = ChatOpenAI(
                model=self._model_name,
                api_key=secret,
                base_url=self._settings.vercel_base_url,
                temperature=self._settings.llm_temperature,
                timeout=self._settings.llm_timeout,
                default_headers=headers,
                model_kwargs=model_kwargs,
            )
            response = client.invoke(langchain_messages)
        except Exception as err:
            # One line — picks LLMAuthError / LLMRateLimitError / etc. automatically.
            raise LLMError.from_provider_error(err, provider=self.PROVIDER) from err

        return extract_text_from_content(
            response.content,
            provider_name=self.PROVIDER,
        )
