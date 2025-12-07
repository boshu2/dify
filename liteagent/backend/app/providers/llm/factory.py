from app.providers.llm.base import BaseLLMProvider
from app.providers.llm.openai_provider import OpenAIProvider
from app.providers.llm.anthropic_provider import AnthropicProvider
from app.providers.llm.ollama_provider import OllamaProvider
from app.schemas.provider import ProviderType


class LLMProviderFactory:
    """Factory for creating LLM provider instances."""

    _providers: dict[ProviderType, type[BaseLLMProvider]] = {
        ProviderType.OPENAI: OpenAIProvider,
        ProviderType.ANTHROPIC: AnthropicProvider,
        ProviderType.OLLAMA: OllamaProvider,
    }

    @classmethod
    def create(
        cls,
        provider_type: ProviderType,
        api_key: str | None,
        model_name: str,
        base_url: str | None = None,
    ) -> BaseLLMProvider:
        """Create an LLM provider instance."""
        provider_class = cls._providers.get(provider_type)
        if not provider_class:
            raise ValueError(f"Unknown provider type: {provider_type}")

        return provider_class(api_key=api_key, model_name=model_name, base_url=base_url)

    @classmethod
    def get_available_models(cls, provider_type: ProviderType) -> list[str]:
        """Get available models for a provider type."""
        provider_class = cls._providers.get(provider_type)
        if not provider_class:
            raise ValueError(f"Unknown provider type: {provider_type}")

        return provider_class.get_available_models()

    @classmethod
    def get_all_provider_types(cls) -> list[ProviderType]:
        """Get all supported provider types."""
        return list(cls._providers.keys())
