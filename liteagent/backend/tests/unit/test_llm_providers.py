"""
Unit tests for LLM providers.
Tests the provider abstraction layer and factory pattern.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.providers.llm.base import BaseLLMProvider, LLMMessage, LLMResponse
from app.providers.llm.factory import LLMProviderFactory
from app.providers.llm.openai_provider import OpenAIProvider
from app.providers.llm.anthropic_provider import AnthropicProvider
from app.providers.llm.ollama_provider import OllamaProvider
from app.schemas.provider import ProviderType


class TestLLMMessage:
    """Tests for LLMMessage dataclass."""

    def test_create_message(self):
        """Test creating an LLM message."""
        msg = LLMMessage(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_create_system_message(self):
        """Test creating a system message."""
        msg = LLMMessage(role="system", content="You are helpful.")
        assert msg.role == "system"
        assert msg.content == "You are helpful."


class TestLLMResponse:
    """Tests for LLMResponse dataclass."""

    def test_create_response(self):
        """Test creating an LLM response."""
        response = LLMResponse(
            content="Hello back!",
            model="gpt-4o",
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )
        assert response.content == "Hello back!"
        assert response.model == "gpt-4o"
        assert response.usage["total_tokens"] == 15

    def test_response_without_usage(self):
        """Test creating a response without usage stats."""
        response = LLMResponse(content="Hello", model="gpt-4o")
        assert response.usage is None


class TestLLMProviderFactory:
    """Tests for LLMProviderFactory."""

    def test_create_openai_provider(self):
        """Test creating an OpenAI provider."""
        provider = LLMProviderFactory.create(
            provider_type=ProviderType.OPENAI,
            api_key="sk-test",
            model_name="gpt-4o",
        )
        assert isinstance(provider, OpenAIProvider)
        assert provider.model_name == "gpt-4o"

    def test_create_anthropic_provider(self):
        """Test creating an Anthropic provider."""
        provider = LLMProviderFactory.create(
            provider_type=ProviderType.ANTHROPIC,
            api_key="sk-ant-test",
            model_name="claude-3-opus-20240229",
        )
        assert isinstance(provider, AnthropicProvider)
        assert provider.model_name == "claude-3-opus-20240229"

    def test_create_ollama_provider(self):
        """Test creating an Ollama provider."""
        provider = LLMProviderFactory.create(
            provider_type=ProviderType.OLLAMA,
            api_key=None,
            model_name="llama3.2",
            base_url="http://localhost:11434",
        )
        assert isinstance(provider, OllamaProvider)
        assert provider.model_name == "llama3.2"
        assert provider.base_url == "http://localhost:11434"

    def test_create_openai_compatible_provider(self):
        """Test creating an OpenAI-compatible provider for custom endpoints."""
        provider = LLMProviderFactory.create(
            provider_type=ProviderType.OPENAI_COMPATIBLE,
            api_key="custom-key",
            model_name="custom-model",
            base_url="https://my-vllm-server.com/v1",
        )
        # This test will fail until we implement OpenAICompatibleProvider
        assert provider is not None
        assert provider.base_url == "https://my-vllm-server.com/v1"
        assert provider.model_name == "custom-model"

    def test_get_available_models_openai(self):
        """Test getting available models for OpenAI."""
        models = LLMProviderFactory.get_available_models(ProviderType.OPENAI)
        assert "gpt-4o" in models
        assert "gpt-4o-mini" in models
        assert "gpt-3.5-turbo" in models

    def test_get_available_models_anthropic(self):
        """Test getting available models for Anthropic."""
        models = LLMProviderFactory.get_available_models(ProviderType.ANTHROPIC)
        assert any("claude" in m for m in models)

    def test_get_all_provider_types(self):
        """Test getting all provider types."""
        types = LLMProviderFactory.get_all_provider_types()
        assert ProviderType.OPENAI in types
        assert ProviderType.ANTHROPIC in types
        assert ProviderType.OLLAMA in types
        # Should also include openai_compatible after implementation
        assert ProviderType.OPENAI_COMPATIBLE in types

    def test_invalid_provider_type_raises_error(self):
        """Test that invalid provider type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown provider type"):
            LLMProviderFactory.create(
                provider_type="invalid_type",
                api_key="test",
                model_name="test",
            )


class TestOpenAIProvider:
    """Tests for OpenAI provider."""

    def test_init(self):
        """Test OpenAI provider initialization."""
        provider = OpenAIProvider(
            api_key="sk-test",
            model_name="gpt-4o",
        )
        assert provider.api_key == "sk-test"
        assert provider.model_name == "gpt-4o"

    def test_init_with_base_url(self):
        """Test OpenAI provider with custom base URL."""
        provider = OpenAIProvider(
            api_key="sk-test",
            model_name="gpt-4o",
            base_url="https://custom-endpoint.com/v1",
        )
        assert provider.base_url == "https://custom-endpoint.com/v1"

    def test_get_available_models(self):
        """Test getting available OpenAI models."""
        models = OpenAIProvider.get_available_models()
        assert len(models) > 0
        assert "gpt-4o" in models

    @pytest.mark.asyncio
    async def test_chat_success(self, mock_openai_response):
        """Test successful chat completion."""
        provider = OpenAIProvider(api_key="sk-test", model_name="gpt-4o")

        with patch.object(provider.client.chat.completions, "create", new_callable=AsyncMock) as mock_create:
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "Test response"
            mock_response.model = "gpt-4o"
            mock_response.usage = MagicMock()
            mock_response.usage.prompt_tokens = 10
            mock_response.usage.completion_tokens = 20
            mock_response.usage.total_tokens = 30
            mock_create.return_value = mock_response

            messages = [LLMMessage(role="user", content="Hello")]
            response = await provider.chat(messages)

            assert response.content == "Test response"
            assert response.model == "gpt-4o"
            assert response.usage["total_tokens"] == 30


class TestOpenAICompatibleProvider:
    """Tests for OpenAI-compatible provider (Azure, vLLM, etc.)."""

    def test_init_with_azure_config(self):
        """Test initialization with Azure OpenAI configuration."""
        from app.providers.llm.openai_compatible_provider import OpenAICompatibleProvider

        provider = OpenAICompatibleProvider(
            api_key="azure-api-key",
            model_name="gpt-4-deployment",
            base_url="https://my-resource.openai.azure.com/openai/deployments/gpt-4",
            api_version="2024-02-01",
            extra_headers={"api-key": "azure-api-key"},
        )
        assert provider.api_key == "azure-api-key"
        assert provider.model_name == "gpt-4-deployment"
        assert provider.api_version == "2024-02-01"

    def test_init_with_vllm_config(self):
        """Test initialization with vLLM configuration."""
        from app.providers.llm.openai_compatible_provider import OpenAICompatibleProvider

        provider = OpenAICompatibleProvider(
            api_key="dummy",  # vLLM often doesn't need a real key
            model_name="meta-llama/Llama-2-7b-chat-hf",
            base_url="http://localhost:8000/v1",
        )
        assert provider.base_url == "http://localhost:8000/v1"
        assert provider.model_name == "meta-llama/Llama-2-7b-chat-hf"

    def test_init_with_litellm_proxy(self):
        """Test initialization with LiteLLM proxy configuration."""
        from app.providers.llm.openai_compatible_provider import OpenAICompatibleProvider

        provider = OpenAICompatibleProvider(
            api_key="sk-litellm-key",
            model_name="azure/gpt-4-deployment",
            base_url="http://localhost:4000/v1",
        )
        assert provider.base_url == "http://localhost:4000/v1"

    @pytest.mark.asyncio
    async def test_chat_with_custom_endpoint(self):
        """Test chat with custom OpenAI-compatible endpoint."""
        from app.providers.llm.openai_compatible_provider import OpenAICompatibleProvider

        provider = OpenAICompatibleProvider(
            api_key="test-key",
            model_name="custom-model",
            base_url="https://custom-endpoint.com/v1",
        )

        with patch.object(provider.client.chat.completions, "create", new_callable=AsyncMock) as mock_create:
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "Custom endpoint response"
            mock_response.model = "custom-model"
            mock_response.usage = MagicMock()
            mock_response.usage.prompt_tokens = 5
            mock_response.usage.completion_tokens = 10
            mock_response.usage.total_tokens = 15
            mock_create.return_value = mock_response

            messages = [LLMMessage(role="user", content="Test")]
            response = await provider.chat(messages)

            assert response.content == "Custom endpoint response"
            mock_create.assert_called_once()

    def test_get_available_models_returns_empty(self):
        """OpenAI-compatible providers don't have predefined models."""
        from app.providers.llm.openai_compatible_provider import OpenAICompatibleProvider

        models = OpenAICompatibleProvider.get_available_models()
        # Returns empty since models are dynamic/user-provided
        assert models == []


class TestAnthropicProvider:
    """Tests for Anthropic provider."""

    def test_init(self):
        """Test Anthropic provider initialization."""
        provider = AnthropicProvider(
            api_key="sk-ant-test",
            model_name="claude-3-opus-20240229",
        )
        assert provider.api_key == "sk-ant-test"
        assert provider.model_name == "claude-3-opus-20240229"

    def test_get_available_models(self):
        """Test getting available Anthropic models."""
        models = AnthropicProvider.get_available_models()
        assert len(models) > 0
        assert any("claude" in m for m in models)


class TestOllamaProvider:
    """Tests for Ollama provider."""

    def test_init_default_url(self):
        """Test Ollama provider with default URL."""
        provider = OllamaProvider(
            api_key=None,
            model_name="llama3.2",
        )
        assert provider.base_url == "http://localhost:11434"

    def test_init_custom_url(self):
        """Test Ollama provider with custom URL."""
        provider = OllamaProvider(
            api_key=None,
            model_name="llama3.2",
            base_url="http://custom-ollama:11434",
        )
        assert provider.base_url == "http://custom-ollama:11434"

    def test_get_available_models(self):
        """Test getting default Ollama models."""
        models = OllamaProvider.get_available_models()
        assert "llama3.2" in models
        assert "mistral" in models
