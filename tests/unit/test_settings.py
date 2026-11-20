from __future__ import annotations

import pytest
from pydantic import ValidationError

from project6_1.core.config.settings import (
    AppSettings,
    get_settings,
    reset_settings_cache,
)
from project6_1.core.dependency.providers import (
    DEFAULT_BASE_URLS,
    DEFAULT_MODELS,
    LLMProviderEnum,
)


def test_default_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in ("LLM_PROVIDER", "LLM_MODEL", "LLM_TEMPERATURE", "OPENAI_API_KEY"):
        monkeypatch.delenv(key, raising=False)

    # `_env_file` is a real pydantic-settings v2 init kwarg used to disable
    # .env loading in tests; the runtime stubs expose it but the type stubs
    # Pylance ships do not, so the call is annotated to suppress the warning.
    settings = AppSettings(_env_file=None)  # type: ignore[call-arg]
    assert settings.llm_provider == LLMProviderEnum.GEMINI
    assert settings.llm_model == "gpt-4o-mini"
    assert settings.llm_temperature == 0.0
    assert settings.llm_max_tokens is None
    assert settings.openai_api_key is None
    assert settings.openai_base_url == DEFAULT_BASE_URLS[LLMProviderEnum.OPENAI]


def test_env_overrides_and_secret_str(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("LLM_MODEL", "openrouter/auto")
    monkeypatch.setenv("LLM_TEMPERATURE", "0.7")
    monkeypatch.setenv("LLM_MAX_TOKENS", "512")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-secret-key")

    settings = AppSettings(_env_file=None)  # type: ignore[call-arg]
    assert settings.llm_provider == LLMProviderEnum.OPENROUTER
    assert settings.llm_model == "openrouter/auto"
    assert settings.llm_temperature == 0.7
    assert settings.llm_max_tokens == 512
    assert settings.openrouter_api_key is not None
    assert settings.openrouter_api_key.get_secret_value() == "sk-test-secret-key"
    assert "sk-test-secret-key" not in str(settings.openrouter_api_key)


def test_invalid_provider_raises_validation_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "invalid-provider")

    with pytest.raises(ValidationError):
        AppSettings(_env_file=None)  # type: ignore[call-arg]


def test_get_settings_caches_and_reset_clears(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_MODEL", "model-v1")
    first = get_settings()
    assert first.llm_model == "model-v1"

    monkeypatch.setenv("LLM_MODEL", "model-v2")
    assert get_settings() is first
    assert get_settings().llm_model == "model-v1"

    reset_settings_cache()
    refreshed = get_settings()
    assert refreshed is not first
    assert refreshed.llm_model == "model-v2"


def test_default_models_exist_for_all_providers() -> None:
    for provider in LLMProviderEnum:
        assert provider in DEFAULT_MODELS
        assert provider in DEFAULT_BASE_URLS
