"""
Embedding providers for RAG pipeline.

Supports:
- NVIDIA Nemotron embeddings (via NIM API)
- No-embedding mode (for BM25/keyword search)
"""
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass

import httpx


@dataclass
class EmbeddingResult:
    """Result from embedding operation."""
    embeddings: list[list[float]]
    model: str
    usage: dict[str, int] | None = None


class EmbeddingProvider(ABC):
    """Base class for embedding providers."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return embedding dimension."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return model name."""
        pass

    @abstractmethod
    async def embed(self, texts: list[str]) -> EmbeddingResult:
        """
        Embed a list of texts.

        Args:
            texts: List of strings to embed.

        Returns:
            EmbeddingResult with embeddings.
        """
        pass

    async def embed_query(self, query: str) -> list[float]:
        """Embed a single query. May use different prompt for retrieval."""
        result = await self.embed([query])
        return result.embeddings[0]

    async def embed_documents(self, documents: list[str]) -> list[list[float]]:
        """Embed multiple documents."""
        result = await self.embed(documents)
        return result.embeddings


class NemotronEmbedder(EmbeddingProvider):
    """
    NVIDIA Nemotron embedding provider.

    Uses the Llama Nemotron Embed model via NVIDIA NIM API.
    Reference: https://huggingface.co/nvidia/llama-nemotron-embed-1b-v2
    """

    # Model variants
    NEMOTRON_1B = "nvidia/llama-nemotron-embed-1b-v2"
    NEMOTRON_8B = "nvidia/llama-embed-nemotron-8b"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://integrate.api.nvidia.com/v1",
        model: str = NEMOTRON_1B,
        timeout: float = 30.0,
    ):
        """
        Initialize Nemotron embedder.

        Args:
            api_key: NVIDIA API key. Defaults to NVIDIA_API_KEY env var.
            base_url: API base URL.
            model: Model to use.
            timeout: Request timeout in seconds.
        """
        self.api_key = api_key or os.environ.get("NVIDIA_API_KEY", "")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

        # Nemotron 1B v2 outputs 4096-dim embeddings
        # Supports Matryoshka (can truncate to 256, 512, 1024, 2048)
        self._dimension = 4096

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self.model

    async def embed(self, texts: list[str]) -> EmbeddingResult:
        """
        Embed texts using Nemotron.

        The model is instruction-aware for retrieval:
        - Queries: "Instruct: {task}\nQuery: {query}"
        - Documents: Just the text
        """
        if not texts:
            return EmbeddingResult(embeddings=[], model=self.model)

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/embeddings",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "input": texts,
                    "encoding_format": "float",
                },
            )
            response.raise_for_status()
            data = response.json()

        embeddings = [item["embedding"] for item in data.get("data", [])]

        return EmbeddingResult(
            embeddings=embeddings,
            model=self.model,
            usage=data.get("usage"),
        )

    async def embed_query(self, query: str) -> list[float]:
        """
        Embed a query with retrieval instruction.

        Nemotron performs better with task instructions for queries.
        """
        # Add retrieval instruction for better performance
        instructed_query = f"Instruct: Given a query, retrieve relevant passages\nQuery: {query}"
        result = await self.embed([instructed_query])
        return result.embeddings[0]


class NoEmbedder(EmbeddingProvider):
    """
    No-op embedder for BM25/keyword-only retrieval.

    Returns empty embeddings - retrieval uses text matching instead.
    """

    def __init__(self):
        self._dimension = 0

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return "none"

    async def embed(self, texts: list[str]) -> EmbeddingResult:
        """Return empty embeddings."""
        return EmbeddingResult(
            embeddings=[[] for _ in texts],
            model="none",
        )


class OpenAIEmbedder(EmbeddingProvider):
    """OpenAI embedding provider for comparison/fallback."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "text-embedding-3-small",
    ):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.model = model

        # Dimension depends on model
        self._dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
        self._dimension = self._dimensions.get(model, 1536)

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self.model

    async def embed(self, texts: list[str]) -> EmbeddingResult:
        """Embed using OpenAI API."""
        if not texts:
            return EmbeddingResult(embeddings=[], model=self.model)

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://api.openai.com/v1/embeddings",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "input": texts,
                },
            )
            response.raise_for_status()
            data = response.json()

        embeddings = [item["embedding"] for item in data.get("data", [])]

        return EmbeddingResult(
            embeddings=embeddings,
            model=self.model,
            usage=data.get("usage"),
        )


def create_embedder(
    provider: str = "nemotron",
    **kwargs,
) -> EmbeddingProvider:
    """
    Factory function to create embedder.

    Args:
        provider: "nemotron", "openai", or "none"
        **kwargs: Provider-specific arguments.

    Returns:
        EmbeddingProvider instance.
    """
    providers = {
        "nemotron": NemotronEmbedder,
        "openai": OpenAIEmbedder,
        "none": NoEmbedder,
    }

    if provider not in providers:
        raise ValueError(f"Unknown embedding provider: {provider}")

    return providers[provider](**kwargs)
