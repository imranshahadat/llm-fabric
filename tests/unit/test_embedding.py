"""Test that the Gemini embedding adapter returns a vector of numbers.

We fake the Google AI SDK so no real API call is made.

Run with:
    pytest tests/unit/test_embedding.py
"""

from __future__ import annotations

import asyncio
import sys
import types

from project6_1.core.config.settings import AppSettings, LLMProviderEnum
from project6_1.domains.infrastructure.embed.gemini_embed import GeminiEmbeddingModel

# ── Fake Google SDK ─────────────────────────────────────────────────────
# This class pretends to be GoogleGenerativeAIEmbeddings.
# Instead of calling Google, it returns a small test vector.


class _StubEmbeddings:
    """Fake version of GoogleGenerativeAIEmbeddings for testing."""

    def __init__(self, **_kwargs: object) -> None:
        """Ignore arguments like model name and API key."""

    def embed_query(self, text: str) -> list[float]:
        """Return a short vector. First value = text length so we can check it."""
        return [float(len(text)), 0.1, 0.2, 0.3]


def _install_stub_sdk() -> None:
    """Put the fake SDK into sys.modules so the import finds it."""
    module: types.ModuleType = types.ModuleType("langchain_google_genai")
    module.GoogleGenerativeAIEmbeddings = _StubEmbeddings  # type: ignore[arg-type]
    sys.modules["langchain_google_genai"] = module


def _remove_stub_sdk() -> None:
    """Remove the fake SDK so it does not affect other tests."""
    sys.modules.pop("langchain_google_genai", None)


# ── The test ────────────────────────────────────────────────────────────


def test_generate_embedding_returns_float_vector() -> None:
    """generate_embedding should return a list of float numbers."""
    # Set up test settings. The API key is fake — no real key needed.
    settings: AppSettings = AppSettings(
        gemini_api_key="test-key-not-real",  # type: ignore[arg-type]
        gemini_embedding_model="gemini-embedding-001",
        embed_provider=LLMProviderEnum.GEMINI,
        llm_provider=LLMProviderEnum.GEMINI,
        llm_announce=False,
    )

    # Install the fake SDK before we create the adapter.
    _install_stub_sdk()
    try:
        adapter: GeminiEmbeddingModel = GeminiEmbeddingModel(settings)

        # "hello" has 5 letters, so the fake returns [5.0, 0.1, 0.2, 0.3].
        vector: list[float] = asyncio.run(adapter.generate_embedding({"text": "hello"}))
    finally:
        # Always clean up so the fake does not leak into other tests.
        _remove_stub_sdk()

    # Check the result.
    assert isinstance(vector, list)
    assert len(vector) == 4
    assert vector[0] == 5.0  # 5 = len("hello") → text was passed through correctly
    assert all(isinstance(value, float) for value in vector)
