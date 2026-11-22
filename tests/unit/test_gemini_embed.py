# tests/unit/test_gemini_embed.py
import sys
from unittest import mock

import pytest

# tests/unit/_helpers.py
from pydantic import SecretStr

from project6_1.core.config.settings import AppSettings
from project6_1.core.exceptions.llm import LLMError
from project6_1.domains.infrastructure.embed.gemini_embed import GeminiEmbeddingModel


def secret(value: str = "test-key") -> SecretStr:
    """Wrap a literal as SecretStr so type checkers stop complaining in tests."""
    return SecretStr(value)


class TestGeminiEmbeddingModelInitialization:
    # ... keep your existing tests for client creation, default model, custom model ...

    def test_raises_import_error_when_sdk_missing(self) -> None:
        """Should raise LLMError with a helpful message when the SDK is not importable."""
        # Setting a sys.modules entry to None forces Python to raise ImportError
        # on the next `import <name>` — see CPython docs on the import system.
        # This is independent of whether the real package or a stub is installed.
        settings = AppSettings(gemini_api_key=secret())

        with (
            mock.patch.dict(sys.modules, {"langchain_google_genai": None}),
            pytest.raises(LLMError) as exc_info,
        ):
            GeminiEmbeddingModel(settings)

        message = str(exc_info.value).lower()
        assert "langchain-google-genai" in message
        assert "uv sync" in message
        assert exc_info.value.provider == "gemini"
