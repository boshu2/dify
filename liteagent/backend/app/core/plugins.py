"""
Plugin Loader - Auto-registers all available providers.

This module bootstraps the provider registries with all built-in providers.
External plugins can also register their providers using the registry decorators.

Usage:
    from app.core.plugins import load_all_plugins
    load_all_plugins()

    # Now all providers are registered
    from app.core.registry import llm_registry
    provider = llm_registry.get("openai")
"""
import logging
from typing import Any

from app.core.registry import (
    llm_registry,
    embedding_registry,
    vector_store_registry,
    storage_registry,
    chunker_registry,
    retriever_registry,
    LLMProvider,
    EmbeddingProvider,
    VectorStoreProvider,
    StorageProvider,
    ChunkerProvider,
)

logger = logging.getLogger(__name__)


def _register_llm_providers() -> None:
    """Register built-in LLM providers."""
    from app.providers.llm.base import BaseLLMProvider, LLMMessage, LLMResponse
    from app.providers.llm.openai_provider import OpenAIProvider
    from app.providers.llm.anthropic_provider import AnthropicProvider
    from app.providers.llm.ollama_provider import OllamaProvider
    from app.providers.llm.openai_compatible_provider import OpenAICompatibleProvider

    # Adapter to make existing providers compatible with registry interface
    class LLMProviderAdapter(LLMProvider):
        """Adapts existing BaseLLMProvider to registry LLMProvider interface."""

        def __init__(self, provider: BaseLLMProvider):
            self._provider = provider

        async def chat(
            self,
            messages: list[dict[str, Any]],
            tools: list[dict[str, Any]] | None = None,
            **kwargs,
        ) -> dict[str, Any]:
            llm_messages = [LLMMessage(role=m["role"], content=m["content"]) for m in messages]
            response = await self._provider.chat(llm_messages)
            return {
                "choices": [{"message": {"content": response.content, "role": "assistant"}}],
                "model": response.model,
                "usage": response.usage,
            }

        async def complete(self, prompt: str, **kwargs) -> str:
            response = await self.chat([{"role": "user", "content": prompt}], **kwargs)
            return response["choices"][0]["message"]["content"]

    # Register factories that create adapted providers
    llm_registry.register_factory(
        "openai",
        lambda api_key, model="gpt-4", **kw: LLMProviderAdapter(
            OpenAIProvider(api_key=api_key, model_name=model, **kw)
        ),
        description="OpenAI GPT models",
        is_default=True,
    )

    llm_registry.register_factory(
        "anthropic",
        lambda api_key, model="claude-3-sonnet-20240229", **kw: LLMProviderAdapter(
            AnthropicProvider(api_key=api_key, model_name=model, **kw)
        ),
        description="Anthropic Claude models",
    )

    llm_registry.register_factory(
        "ollama",
        lambda model="llama2", base_url="http://localhost:11434", **kw: LLMProviderAdapter(
            OllamaProvider(api_key=None, model_name=model, base_url=base_url, **kw)
        ),
        description="Local Ollama models",
    )

    llm_registry.register_factory(
        "openai-compatible",
        lambda api_key, model, base_url, **kw: LLMProviderAdapter(
            OpenAICompatibleProvider(api_key=api_key, model_name=model, base_url=base_url, **kw)
        ),
        description="OpenAI-compatible API endpoints",
    )

    logger.debug("Registered LLM providers: openai, anthropic, ollama, openai-compatible")


def _register_embedding_providers() -> None:
    """Register built-in embedding providers."""
    from app.rag.embeddings import (
        EmbeddingProvider as RagEmbeddingProvider,
        NemotronEmbedder,
        OpenAIEmbedder,
        NoEmbedder,
    )

    # Adapter for existing embedders
    class EmbeddingAdapter(EmbeddingProvider):
        def __init__(self, provider: RagEmbeddingProvider):
            self._provider = provider

        async def embed(self, texts: list[str]) -> list[list[float]]:
            result = await self._provider.embed(texts)
            return result.embeddings

        def get_dimension(self) -> int:
            return self._provider.dimension

    embedding_registry.register_factory(
        "nemotron",
        lambda **kw: EmbeddingAdapter(NemotronEmbedder(**kw)),
        description="NVIDIA Nemotron embeddings",
        is_default=True,
    )

    embedding_registry.register_factory(
        "openai",
        lambda **kw: EmbeddingAdapter(OpenAIEmbedder(**kw)),
        description="OpenAI embeddings",
    )

    embedding_registry.register_factory(
        "none",
        lambda: EmbeddingAdapter(NoEmbedder()),
        description="No embeddings (for keyword search)",
    )

    logger.debug("Registered embedding providers: nemotron, openai, none")


def _register_vector_store_providers() -> None:
    """Register built-in vector store providers."""
    from app.rag.vector_store import (
        VectorStore,
        PgVectorStore,
        InMemoryVectorStore,
    )

    # Adapter for existing vector stores
    class VectorStoreAdapter(VectorStoreProvider):
        def __init__(self, store: VectorStore):
            self._store = store

        async def add(
            self,
            vectors: list[list[float]],
            ids: list[str],
            metadata: list[dict[str, Any]] | None = None,
        ) -> None:
            from app.rag.vector_store import Document
            docs = []
            for i, (vec, doc_id) in enumerate(zip(vectors, ids)):
                meta = metadata[i] if metadata else {}
                docs.append(Document(id=doc_id, content="", embedding=vec, metadata=meta))
            await self._store.add_documents(docs)

        async def search(
            self,
            query_vector: list[float],
            top_k: int = 10,
            filter: dict[str, Any] | None = None,
        ) -> list[dict[str, Any]]:
            results = await self._store.search(query_vector, limit=top_k, filter=filter)
            return [
                {
                    "id": r.document.id,
                    "score": r.score,
                    "content": r.document.content,
                    "metadata": r.document.metadata,
                }
                for r in results
            ]

        async def delete(self, ids: list[str]) -> None:
            await self._store.delete(ids)

    vector_store_registry.register_factory(
        "memory",
        lambda: VectorStoreAdapter(InMemoryVectorStore()),
        description="In-memory vector store (for testing)",
        is_default=True,
    )

    vector_store_registry.register_factory(
        "pgvector",
        lambda connection_string, **kw: VectorStoreAdapter(
            PgVectorStore(connection_string=connection_string, **kw)
        ),
        description="PostgreSQL with pgvector extension",
    )

    logger.debug("Registered vector store providers: memory, pgvector")


def _register_storage_providers() -> None:
    """Register built-in storage providers."""

    class MemoryStorageProvider(StorageProvider):
        """In-memory key-value storage."""

        def __init__(self):
            self._data: dict[str, bytes] = {}

        async def get(self, key: str) -> bytes | None:
            return self._data.get(key)

        async def set(self, key: str, value: bytes, ttl: int | None = None) -> None:
            self._data[key] = value

        async def delete(self, key: str) -> None:
            self._data.pop(key, None)

        async def exists(self, key: str) -> bool:
            return key in self._data

    class RedisStorageProvider(StorageProvider):
        """Redis-backed storage."""

        def __init__(self):
            from app.core.redis_client import redis_client
            self._client = redis_client

        async def get(self, key: str) -> bytes | None:
            return self._client.get(key)

        async def set(self, key: str, value: bytes, ttl: int | None = None) -> None:
            if ttl:
                self._client.setex(key, ttl, value)
            else:
                self._client.set(key, value)

        async def delete(self, key: str) -> None:
            self._client.delete(key)

        async def exists(self, key: str) -> bool:
            return bool(self._client.exists(key))

    storage_registry.register_factory(
        "memory",
        MemoryStorageProvider,
        description="In-memory storage",
        is_default=True,
    )

    storage_registry.register_factory(
        "redis",
        RedisStorageProvider,
        description="Redis storage",
    )

    logger.debug("Registered storage providers: memory, redis")


def _register_chunker_providers() -> None:
    """Register built-in chunker providers."""
    from app.rag.chunker import (
        TextChunker,
        FixedSizeChunker,
        RecursiveChunker,
        SemanticChunker,
    )

    class ChunkerAdapter(ChunkerProvider):
        def __init__(self, chunker: TextChunker):
            self._chunker = chunker

        def chunk(self, text: str, **kwargs) -> list[str]:
            chunks = self._chunker.chunk(text)
            return [c.text for c in chunks]

    chunker_registry.register_factory(
        "fixed",
        lambda chunk_size=1000, overlap=200: ChunkerAdapter(
            FixedSizeChunker(chunk_size=chunk_size, overlap=overlap)
        ),
        description="Fixed-size text splitter",
        is_default=True,
    )

    chunker_registry.register_factory(
        "recursive",
        lambda chunk_size=1000, overlap=200: ChunkerAdapter(
            RecursiveChunker(chunk_size=chunk_size, overlap=overlap)
        ),
        description="Recursive character text splitter",
    )

    chunker_registry.register_factory(
        "semantic",
        lambda **kw: ChunkerAdapter(
            SemanticChunker(**kw)
        ),
        description="Semantic similarity splitting",
    )

    logger.debug("Registered chunker providers: fixed, recursive, semantic")


_loaded = False


def load_all_plugins() -> None:
    """Load and register all built-in plugins."""
    global _loaded
    if _loaded:
        return

    logger.info("Loading plugin providers...")

    _register_llm_providers()
    _register_embedding_providers()
    _register_vector_store_providers()
    _register_storage_providers()
    _register_chunker_providers()

    _loaded = True
    logger.info("All plugins loaded successfully")


def get_provider_summary() -> dict[str, list[str]]:
    """Get summary of all registered providers."""
    from app.core.registry import list_providers
    return list_providers()
