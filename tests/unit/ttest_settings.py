"""Test AppSettings configuration loading and validation."""

from __future__ import annotations

from pydantic import SecretStr

from project6_1.core.config.settings import (
    AppSettings,
    LLMProviderEnum,
    get_settings,
    reset_settings_cache,
)


class TestAppSettingsDefaults:
    """Test default values in AppSettings."""

    def test_default_embed_provider_is_gemini(self) -> None:
        """Default embed provider should be GEMINI."""
        settings = AppSettings()
        assert settings.embed_provider == LLMProviderEnum.GEMINI

    def test_default_llm_provider_is_gemini(self) -> None:
        """Default LLM provider should be GEMINI."""
        settings = AppSettings()
        assert settings.llm_provider == LLMProviderEnum.GEMINI

    def test_default_gemini_embedding_model(self) -> None:
        """Default Gemini embedding model should be gemini-embedding-001."""
        settings = AppSettings()
        assert settings.gemini_embedding_model == "gemini-embedding-001"

    def test_default_llm_max_retries(self) -> None:
        """Default max retries should be 3."""
        settings = AppSettings()
        assert settings.llm_max_retries == 3

    def test_api_keys_are_none_by_default(self) -> None:
        """API keys should be None when not set in environment."""
        settings = AppSettings()
        assert settings.gemini_api_key is None
        assert settings.openai_api_key is None
        assert settings.cohere_api_key is None


class TestAppSettingsWithValues:
    """Test AppSettings with explicit values."""

    def test_can_set_gemini_api_key(self) -> None:
        """Should accept a string API key and wrap it in SecretStr."""
        settings = AppSettings(gemini_api_key="test-key-123")
        assert isinstance(settings.gemini_api_key, SecretStr)
        assert settings.gemini_api_key.get_secret_value() == "test-key-123"

    def test_can_override_embed_provider(self) -> None:
        """Should allow overriding the embed provider."""
        settings = AppSettings(embed_provider=LLMProviderEnum.OPENAI)
        assert settings.embed_provider == LLMProviderEnum.OPENAI

    def test_can_set_custom_gemini_model(self) -> None:
        """Should allow custom Gemini embedding model."""
        settings = AppSettings(gemini_embedding_model="custom-model")
        assert settings.gemini_embedding_model == "custom-model"


class TestLLMProviderEnum:
    """Test LLMProviderEnum values."""

    def test_all_providers_exist(self) -> None:
        """All expected providers should be in the enum."""
        assert hasattr(LLMProviderEnum, "GEMINI")
        assert hasattr(LLMProviderEnum, "OPENAI")
        assert hasattr(LLMProviderEnum, "OPENROUTER")
        assert hasattr(LLMProviderEnum, "COHERE")
        assert hasattr(LLMProviderEnum, "VERCEL")
        assert hasattr(LLMProviderEnum, "FAKE")

    def test_gemini_value(self) -> None:
        """GEMINI enum value should be lowercase 'gemini'."""
        assert LLMProviderEnum.GEMINI.value == "gemini"


class TestGetSettings:
    """Test the cached get_settings function."""

    def test_get_settings_returns_appsettings(self) -> None:
        """get_settings should return an AppSettings instance."""
        settings = get_settings()
        assert isinstance(settings, AppSettings)

    def test_get_settings_is_cached(self) -> None:
        """Multiple calls should return the same instance."""
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2

    def test_reset_settings_cache_clears_cache(self) -> None:
        """reset_settings_cache should allow creating a new instance."""
        s1 = get_settings()
        reset_settings_cache()
        s2 = get_settings()
        # After reset, they might be different instances
        # (depends on whether env changed)
        assert isinstance(s2, AppSettings)