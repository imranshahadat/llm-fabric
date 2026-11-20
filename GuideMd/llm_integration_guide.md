# LLM Integration & Architecture Guide

This guide explains the architectural patterns used for the LLM layer in this project, which files to create or update when integrating a new LLM provider, and how to write unit tests for your integration.

---

## 1. Architectural Patterns Used

This LLM module combines four standard software design patterns so that application code stays completely decoupled from third-party LLM SDKs:

1. **Ports & Adapters (Hexagonal Architecture)**
   * **Port (`BaseLLM`):** Defines the contract (`generate(messages) -> str`) that the rest of the application depends on.
   * **Adapters (`GeminiLLM`, `VercelLLM`, etc.):** Concrete implementations that translate the internal `Message` format into provider-specific SDK calls and map SDK exceptions into unified `LLMError` types.
2. **Template Method (`BaseLangChainLLM`)**
   * Owns cross-cutting concerns—structured logging (`structlog`), latency measurement, thread offloading (`asyncio.to_thread`), and exponential backoff retries (`with_retry`).
   * Subclasses only implement `_generate()` and `_resolve_model_name()`.
3. **Factory & Registry (`factory.py` + `_builder.py`)**
   * Maps string provider names to adapter classes (`LLM_REGISTRY`) and validates required API keys (`API_KEY_REQUIREMENTS`) before instantiation.
4. **Composition Root / Singleton Cache (`get_llm()`)**
   * Instantiates and caches the active `BaseLLM` client once per process using `@lru_cache(maxsize=1)`.

---

## 2. Files to Create & Update for a New Provider

Suppose you want to integrate a brand-new provider (for example, **Cohere** or **Vercel AI Gateway**) that is not yet in the codebase. You will create **1 new file** and edit **3 existing files**. You do **not** need to change any business logic, routes, retry helpers, or loggers.

| Step | Action | File Path | Purpose |
| :--- | :--- | :--- | :--- |
| **1** | **Create** | `domains/infrastructure/llm/<provider>_chat.py` | Implement the concrete adapter subclassing `BaseLangChainLLM`. |
| **2** | **Edit** | `core/config/settings.py` | Add the provider enum value, API key field, and default model field. |
| **3** | **Edit** | `domains/infrastructure/llm/factory.py` | Register the adapter class and its API key path. |
| **4** | **Edit** | Composition root (`get_llm()` module) | Add the provider's default base URL and model name. |

---

## 3. Step-by-Step Integration Example (Vercel AI Gateway)

### Step 1: Create the Adapter File
Create `domains/infrastructure/llm/vercel_chat.py`:

```python
"""Vercel AI Gateway LLM adapter using LangChain."""

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


class VercelLLM(BaseLangChainLLM):
    """Handles chat completions through the Vercel AI Gateway."""

    PROVIDER = "vercel"
    DEFAULT_MODEL = "openai/gpt-4o-mini"

    def _resolve_model_name(self) -> str:
        return self._settings.vercel_model or self.DEFAULT_MODEL

    def _generate(self, messages: list[Message]) -> str:
        key = self._settings.vercel_api_key
        api_key = key.get_secret_value() if key else None
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:  # pragma: no cover
            raise LLMError(
                "langchain-openai is not installed; run `uv add langchain-openai`",
                provider=self.PROVIDER,
            ) from exc

        lc_messages = [_to_lc(m) for m in messages]
        try:
            client = ChatOpenAI(
                model=self._model_name,
                api_key=api_key,
                base_url=self._settings.vercel_base_url,
                temperature=self._settings.llm_temperature,
                max_tokens=self._settings.llm_max_tokens,
                timeout=self._settings.llm_timeout,
            )
            result = client.invoke(lc_messages)
        except Exception as exc:
            raise _map_vercel_error(exc, self.PROVIDER) from exc

        return _content_to_str(result.content, provider=self.PROVIDER)


def _to_lc(m: Message) -> Any:
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
    """Convert a string or list of content blocks into plain text."""
    if isinstance(content, str):
        if not content:
            raise LLMResponseParseError("empty response content", provider=provider)
        return content

    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif block.get("type") == "text" and isinstance(block.get("text"), str):
            parts.append(block["text"])
        else:
            parts.append(str(block))

    text = "".join(parts)
    if not text:
        raise LLMResponseParseError("empty response content", provider=provider)
    return text


def _map_vercel_error(exc: Exception, provider: str) -> LLMError:
    name = type(exc).__name__
    msg = str(exc).lower()
    if "401" in msg or "403" in msg or "api key" in msg or "unauthorized" in msg:
        return LLMAuthError(str(exc), provider=provider)
    if "429" in msg or "rate" in name.lower() or "quota" in msg:
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
```

---

### Step 2: Update Application Settings (`core/config/settings.py`)
Add the provider to `LLMProviderEnum` and define its settings fields on `AppSettings`:

```python
class LLMProviderEnum(str, Enum):
    FAKE = "fake"
    OPENAI = "openai"
    OPENROUTER = "openrouter"
    GEMINI = "gemini"
    VERCEL = "vercel"


class AppSettings(BaseSettings):
    # ... existing fields ...
    vercel_api_key: SecretStr | None = None
    vercel_model: str = "openai/gpt-4o-mini"
    vercel_base_url: str = "https://ai-gateway.vercel.sh/v1"
```

---

### Step 3: Register the Provider in `domains/infrastructure/llm/factory.py`
Import the adapter and add it to both registries:

```python
from .gemini_chat import GeminiLLM
from .vercel_chat import VercelLLM

LLM_REGISTRY: dict[str, type[BaseLLM]] = {
    "gemini": GeminiLLM,
    "vercel": VercelLLM,
}

API_KEY_REQUIREMENTS: dict[str, str] = {
    "gemini": "gemini_api_key",
    "vercel": "vercel_api_key",
}
```

---

### Step 4: Update Defaults in Composition Root (`get_llm()` file)
Add the default base URL and model identifier:

```python
DEFAULT_BASE_URLS: dict[LLMProviderEnum, str] = {
    LLMProviderEnum.FAKE: "",
    LLMProviderEnum.OPENAI: "https://api.openai.com/v1",
    LLMProviderEnum.OPENROUTER: "https://openrouter.ai/api/v1",
    LLMProviderEnum.GEMINI: "",
    LLMProviderEnum.VERCEL: "https://ai-gateway.vercel.sh/v1",
}

DEFAULT_MODELS: dict[LLMProviderEnum, str] = {
    LLMProviderEnum.FAKE: "fake",
    LLMProviderEnum.OPENAI: "gpt-4o-mini",
    LLMProviderEnum.OPENROUTER: "openai/gpt-4o-mini",
    LLMProviderEnum.GEMINI: "gemini-1.5-flash",
    LLMProviderEnum.VERCEL: "openai/gpt-4o-mini",
}
```

---

### Step 5: Configure `.env` to Switch Providers
Switch the active LLM without modifying code:

```env
LLM_PROVIDER="vercel"
VERCEL_API_KEY="your_vercel_api_key_here"
# Optional: override the default model
LLM_MODEL="openai/gpt-4o-mini"
```

---

## 4. Pytest Unit Test Suite

Create `tests/domains/infrastructure/llm/test_vercel_llm.py` to test factory wiring, missing API key validation, message formatting, retries, and error mapping without making live network calls.

```python
"""Unit tests for the Vercel LLM adapter and factory integration."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from pydantic import SecretStr

from project6_1.core.config.settings import AppSettings, LLMProviderEnum
from project6_1.core.exceptions.base import APIKeyMissingError
from project6_1.core.exceptions.llm import LLMAuthError, LLMRateLimitError
from project6_1.domains.infrastructure.llm.factory import build_llm
from project6_1.domains.infrastructure.llm.vercel_chat import VercelLLM


def _make_settings(**overrides: object) -> AppSettings:
    defaults: dict[str, object] = {
        "llm_provider": LLMProviderEnum.VERCEL,
        "vercel_api_key": SecretStr("test-vercel-key"),
        "vercel_model": "openai/gpt-4o-mini",
        "vercel_base_url": "https://ai-gateway.vercel.sh/v1",
        "llm_max_retries": 2,
    }
    defaults.update(overrides)
    return AppSettings(**defaults)  # type: ignore[arg-type]


def test_build_llm_returns_vercel_adapter() -> None:
    settings = _make_settings()
    llm = build_llm(settings)
    assert isinstance(llm, VercelLLM)


def test_build_llm_raises_when_api_key_missing() -> None:
    settings = _make_settings(vercel_api_key=None)
    with pytest.raises(APIKeyMissingError, match="vercel_api_key"):
        build_llm(settings)


@pytest.mark.asyncio
async def test_generate_success() -> None:
    settings = _make_settings()
    llm = VercelLLM(settings)

    mock_client = MagicMock()
    mock_client.invoke.return_value = SimpleNamespace(content="Hello from Vercel!")

    with patch("langchain_openai.ChatOpenAI", return_value=mock_client):
        reply = await llm.generate(
            [
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "Say hi"},
            ]
        )

    assert reply == "Hello from Vercel!"
    mock_client.invoke.assert_called_once()


@pytest.mark.asyncio
async def test_generate_retries_on_rate_limit() -> None:
    settings = _make_settings(llm_max_retries=2)
    llm = VercelLLM(settings)

    mock_client = MagicMock()
    mock_client.invoke.side_effect = [
        RuntimeError("429 Too Many Requests"),
        SimpleNamespace(content="Recovered after retry"),
    ]

    with patch("langchain_openai.ChatOpenAI", return_value=mock_client):
        reply = await llm.generate([{"role": "user", "content": "Ping"}])

    assert reply == "Recovered after retry"
    assert mock_client.invoke.call_count == 2


@pytest.mark.asyncio
async def test_generate_does_not_retry_auth_error() -> None:
    settings = _make_settings(llm_max_retries=3)
    llm = VercelLLM(settings)

    mock_client = MagicMock()
    mock_client.invoke.side_effect = RuntimeError("401 Unauthorized: invalid api key")

    with patch("langchain_openai.ChatOpenAI", return_value=mock_client):
        with pytest.raises(LLMAuthError):
            await llm.generate([{"role": "user", "content": "Ping"}])

    # Auth errors are non-retryable, so invoke must only be called once
    assert mock_client.invoke.call_count == 1
```
