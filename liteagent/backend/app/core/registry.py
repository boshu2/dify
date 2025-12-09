"""
Provider Registry Pattern for Pluggable Architecture.

Enables runtime registration and discovery of providers for:
- LLM backends (OpenAI, Anthropic, local models)
- Vector stores (FAISS, Pinecone, Weaviate, Qdrant)
- Embedding providers (OpenAI, Cohere, local)
- Storage backends (Redis, memory, SQL)
- Custom workflow nodes

Usage:
    # Register a provider
    @llm_registry.register("openai")
    class OpenAIProvider(LLMProvider):
        ...

    # Get a provider
    provider = llm_registry.get("openai")

    # List available providers
    providers = llm_registry.list()
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Generic, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class ProviderMetadata:
    """Metadata about a registered provider."""

    name: str
    description: str = ""
    version: str = "1.0.0"
    author: str = ""
    config_schema: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    is_default: bool = False


class ProviderRegistry(Generic[T]):
    """
    Generic provider registry for pluggable components.

    Supports:
    - Registration via decorator or method
    - Provider metadata and discovery
    - Default provider selection
    - Lazy initialization
    - Configuration validation
    """

    def __init__(self, name: str):
        self.name = name
        self._providers: dict[str, type[T]] = {}
        self._metadata: dict[str, ProviderMetadata] = {}
        self._instances: dict[str, T] = {}
        self._default: str | None = None
        self._factories: dict[str, Callable[..., T]] = {}

    def register(
        self,
        name: str,
        description: str = "",
        version: str = "1.0.0",
        is_default: bool = False,
        config_schema: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> Callable[[type[T]], type[T]]:
        """
        Decorator to register a provider class.

        Usage:
            @registry.register("my_provider", description="My provider")
            class MyProvider(BaseProvider):
                ...
        """

        def decorator(cls: type[T]) -> type[T]:
            self._providers[name] = cls
            self._metadata[name] = ProviderMetadata(
                name=name,
                description=description,
                version=version,
                config_schema=config_schema or {},
                tags=tags or [],
                is_default=is_default,
            )

            if is_default or self._default is None:
                self._default = name

            logger.debug(f"Registered {self.name} provider: {name}")
            return cls

        return decorator

    def register_factory(
        self,
        name: str,
        factory: Callable[..., T],
        description: str = "",
        is_default: bool = False,
    ) -> None:
        """Register a factory function for lazy instantiation."""
        self._factories[name] = factory
        self._metadata[name] = ProviderMetadata(
            name=name,
            description=description,
            is_default=is_default,
        )

        if is_default or self._default is None:
            self._default = name

    def register_instance(self, name: str, instance: T) -> None:
        """Register a pre-instantiated provider."""
        self._instances[name] = instance
        if name not in self._metadata:
            self._metadata[name] = ProviderMetadata(name=name)

    def get(self, name: str | None = None, **config) -> T:
        """
        Get a provider instance by name.

        Args:
            name: Provider name (uses default if None)
            **config: Configuration passed to provider constructor

        Returns:
            Provider instance

        Raises:
            KeyError: If provider not found
        """
        name = name or self._default

        if name is None:
            raise KeyError(f"No {self.name} providers registered")

        # Check for pre-instantiated
        if name in self._instances:
            return self._instances[name]

        # Check for factory
        if name in self._factories:
            instance = self._factories[name](**config)
            if not config:  # Cache only if no config
                self._instances[name] = instance
            return instance

        # Check for class
        if name in self._providers:
            instance = self._providers[name](**config)
            if not config:  # Cache only if no config
                self._instances[name] = instance
            return instance

        available = ", ".join(self.list())
        raise KeyError(
            f"Unknown {self.name} provider: {name}. Available: {available}"
        )

    def get_class(self, name: str) -> type[T] | None:
        """Get provider class without instantiating."""
        return self._providers.get(name)

    def list_providers(self) -> list[str]:
        """List all registered provider names."""
        all_names = set(self._providers.keys())
        all_names.update(self._factories.keys())
        all_names.update(self._instances.keys())
        return sorted(all_names)

    # Alias for backward compatibility
    list = list_providers

    def get_metadata(self, name: str) -> ProviderMetadata | None:
        """Get provider metadata."""
        return self._metadata.get(name)

    def list_with_metadata(self) -> list[ProviderMetadata]:
        """List all providers with their metadata."""
        return [
            self._metadata.get(name, ProviderMetadata(name=name))
            for name in self.list_providers()
        ]

    def get_default(self) -> str | None:
        """Get default provider name."""
        return self._default

    def set_default(self, name: str) -> None:
        """Set default provider."""
        if name not in self.list():
            raise KeyError(f"Unknown provider: {name}")
        self._default = name

    def has(self, name: str) -> bool:
        """Check if provider is registered."""
        return name in self._providers or name in self._factories or name in self._instances

    def unregister(self, name: str) -> None:
        """Unregister a provider."""
        self._providers.pop(name, None)
        self._factories.pop(name, None)
        self._instances.pop(name, None)
        self._metadata.pop(name, None)

        if self._default == name:
            self._default = next(iter(self.list()), None)

    def clear(self) -> None:
        """Clear all registered providers."""
        self._providers.clear()
        self._factories.clear()
        self._instances.clear()
        self._metadata.clear()
        self._default = None


# =============================================================================
# Base Provider Interfaces
# =============================================================================


class LLMProvider(ABC):
    """Base interface for LLM providers."""

    @abstractmethod
    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Send chat completion request."""
        pass

    @abstractmethod
    async def complete(self, prompt: str, **kwargs) -> str:
        """Simple text completion."""
        pass

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the model."""
        return {}


class EmbeddingProvider(ABC):
    """Base interface for embedding providers."""

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for texts."""
        pass

    @abstractmethod
    def get_dimension(self) -> int:
        """Get embedding dimension."""
        pass

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the model."""
        return {}


class VectorStoreProvider(ABC):
    """Base interface for vector store providers."""

    @abstractmethod
    async def add(
        self,
        vectors: list[list[float]],
        ids: list[str],
        metadata: list[dict[str, Any]] | None = None,
    ) -> None:
        """Add vectors to the store."""
        pass

    @abstractmethod
    async def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Search for similar vectors."""
        pass

    @abstractmethod
    async def delete(self, ids: list[str]) -> None:
        """Delete vectors by ID."""
        pass

    async def get(self, ids: list[str]) -> list[dict[str, Any]]:
        """Get vectors by ID."""
        return []


class StorageProvider(ABC):
    """Base interface for key-value storage providers."""

    @abstractmethod
    async def get(self, key: str) -> bytes | None:
        """Get value by key."""
        pass

    @abstractmethod
    async def set(self, key: str, value: bytes, ttl: int | None = None) -> None:
        """Set value with optional TTL."""
        pass

    @abstractmethod
    async def delete(self, key: str) -> None:
        """Delete key."""
        pass

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        pass


class ChunkerProvider(ABC):
    """Base interface for document chunking providers."""

    @abstractmethod
    def chunk(self, text: str, **kwargs) -> list[str]:
        """Split text into chunks."""
        pass


class RetrieverProvider(ABC):
    """Base interface for retrieval providers."""

    @abstractmethod
    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        **kwargs,
    ) -> list[dict[str, Any]]:
        """Retrieve relevant documents."""
        pass


# =============================================================================
# Global Registries
# =============================================================================

llm_registry: ProviderRegistry[LLMProvider] = ProviderRegistry("llm")
embedding_registry: ProviderRegistry[EmbeddingProvider] = ProviderRegistry("embedding")
vector_store_registry: ProviderRegistry[VectorStoreProvider] = ProviderRegistry("vector_store")
storage_registry: ProviderRegistry[StorageProvider] = ProviderRegistry("storage")
chunker_registry: ProviderRegistry[ChunkerProvider] = ProviderRegistry("chunker")
retriever_registry: ProviderRegistry[RetrieverProvider] = ProviderRegistry("retriever")


def get_provider(registry_name: str, provider_name: str | None = None, **config) -> Any:
    """
    Get a provider from any registry by name.

    Args:
        registry_name: Name of the registry (llm, embedding, vector_store, etc.)
        provider_name: Name of the provider (uses default if None)
        **config: Configuration passed to provider

    Returns:
        Provider instance
    """
    registries = {
        "llm": llm_registry,
        "embedding": embedding_registry,
        "vector_store": vector_store_registry,
        "storage": storage_registry,
        "chunker": chunker_registry,
        "retriever": retriever_registry,
    }

    registry = registries.get(registry_name)
    if registry is None:
        raise KeyError(f"Unknown registry: {registry_name}")

    return registry.get(provider_name, **config)


def list_providers(registry_name: str | None = None) -> dict[str, list[str]]:
    """
    List available providers.

    Args:
        registry_name: Specific registry to list (all if None)

    Returns:
        Dict of registry name to provider names
    """
    registries = {
        "llm": llm_registry,
        "embedding": embedding_registry,
        "vector_store": vector_store_registry,
        "storage": storage_registry,
        "chunker": chunker_registry,
        "retriever": retriever_registry,
    }

    if registry_name:
        registry = registries.get(registry_name)
        if registry is None:
            raise KeyError(f"Unknown registry: {registry_name}")
        return {registry_name: registry.list()}

    return {name: reg.list() for name, reg in registries.items()}
