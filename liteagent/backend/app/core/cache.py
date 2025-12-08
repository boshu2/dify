"""
Caching layer for LiteAgent.
Provides in-memory and Redis-compatible caching for LLM responses.
"""
import os
import json
import hashlib
import time
import asyncio
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any


@dataclass
class CacheConfig:
    """Configuration for caching."""

    default_ttl_seconds: int = 3600  # 1 hour
    max_size: int = 1000
    enable_compression: bool = False
    redis_url: str | None = None

    @classmethod
    def from_env(cls) -> "CacheConfig":
        """Create configuration from environment variables."""
        return cls(
            default_ttl_seconds=int(os.environ.get("CACHE_DEFAULT_TTL", "3600")),
            max_size=int(os.environ.get("CACHE_MAX_SIZE", "1000")),
            enable_compression=os.environ.get("CACHE_ENABLE_COMPRESSION", "").lower() == "true",
            redis_url=os.environ.get("REDIS_URL"),
        )


@dataclass
class CacheKey:
    """Structured cache key."""

    prefix: str
    provider: str
    model: str
    content_hash: str

    def __str__(self) -> str:
        """Convert to string key."""
        return f"{self.prefix}:{self.provider}:{self.model}:{self.content_hash}"


@dataclass
class CacheEntry:
    """A cached value with metadata."""

    value: Any
    created_at: float
    ttl_seconds: int

    @property
    def is_expired(self) -> bool:
        """Check if entry has expired."""
        return time.monotonic() - self.created_at > self.ttl_seconds


def cache_key_from_messages(
    provider: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float = 0.0,
) -> str:
    """
    Generate a cache key from chat messages.

    Args:
        provider: LLM provider name.
        model: Model name.
        messages: List of message dictionaries.
        temperature: Temperature setting (affects output).

    Returns:
        Cache key string.
    """
    # Create deterministic hash of messages
    content = json.dumps(
        {
            "messages": messages,
            "temperature": temperature,
        },
        sort_keys=True,
    )
    content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]

    key = CacheKey(
        prefix="llm",
        provider=provider,
        model=model,
        content_hash=content_hash,
    )
    return str(key)


class InMemoryCache:
    """
    In-memory LRU cache.

    Thread-safe cache with TTL support and LRU eviction.
    """

    def __init__(self, config: CacheConfig | None = None):
        self.config = config or CacheConfig()
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = asyncio.Lock()
        self._hits = 0
        self._misses = 0

    async def get(self, key: str) -> Any | None:
        """
        Get a value from cache.

        Args:
            key: Cache key.

        Returns:
            Cached value or None if not found/expired.
        """
        async with self._lock:
            entry = self._cache.get(key)

            if entry is None:
                self._misses += 1
                return None

            if entry.is_expired:
                del self._cache[key]
                self._misses += 1
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self._hits += 1
            return entry.value

    async def set(
        self,
        key: str,
        value: Any,
        ttl_seconds: int | None = None,
    ) -> None:
        """
        Set a value in cache.

        Args:
            key: Cache key.
            value: Value to cache.
            ttl_seconds: Optional TTL override.
        """
        async with self._lock:
            # Evict oldest if at max size
            while len(self._cache) >= self.config.max_size:
                self._cache.popitem(last=False)

            self._cache[key] = CacheEntry(
                value=value,
                created_at=time.monotonic(),
                ttl_seconds=ttl_seconds or self.config.default_ttl_seconds,
            )
            self._cache.move_to_end(key)

    async def delete(self, key: str) -> bool:
        """
        Delete a key from cache.

        Args:
            key: Cache key.

        Returns:
            True if key existed.
        """
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    async def exists(self, key: str) -> bool:
        """
        Check if key exists and is not expired.

        Args:
            key: Cache key.

        Returns:
            True if key exists and is valid.
        """
        async with self._lock:
            entry = self._cache.get(key)
            if entry is None or entry.is_expired:
                return False
            return True

    async def clear(self) -> None:
        """Clear all cached entries."""
        async with self._lock:
            self._cache.clear()

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        return {
            "size": len(self._cache),
            "max_size": self.config.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": (
                self._hits / (self._hits + self._misses)
                if (self._hits + self._misses) > 0
                else 0
            ),
        }


class LLMResponseCache:
    """
    Specialized cache for LLM responses.

    Caches responses based on provider, model, and message content.
    """

    def __init__(self, config: CacheConfig | None = None):
        self.config = config or CacheConfig()
        self._cache = InMemoryCache(config)

    async def get_cached_response(
        self,
        provider: str,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
    ) -> dict[str, Any] | None:
        """
        Get a cached LLM response.

        Args:
            provider: LLM provider name.
            model: Model name.
            messages: Chat messages.
            temperature: Temperature setting.

        Returns:
            Cached response or None.
        """
        key = cache_key_from_messages(provider, model, messages, temperature)
        return await self._cache.get(key)

    async def cache_response(
        self,
        provider: str,
        model: str,
        messages: list[dict[str, str]],
        response: dict[str, Any],
        temperature: float = 0.0,
        ttl_seconds: int | None = None,
    ) -> None:
        """
        Cache an LLM response.

        Args:
            provider: LLM provider name.
            model: Model name.
            messages: Chat messages.
            response: Response to cache.
            temperature: Temperature setting.
            ttl_seconds: Optional TTL override.
        """
        key = cache_key_from_messages(provider, model, messages, temperature)
        await self._cache.set(key, response, ttl_seconds)

    async def invalidate(
        self,
        provider: str,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
    ) -> bool:
        """
        Invalidate a cached response.

        Args:
            provider: LLM provider name.
            model: Model name.
            messages: Chat messages.
            temperature: Temperature setting.

        Returns:
            True if entry was invalidated.
        """
        key = cache_key_from_messages(provider, model, messages, temperature)
        return await self._cache.delete(key)

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        return self._cache.get_stats()


# Global cache instance
_llm_cache: LLMResponseCache | None = None


def get_llm_cache() -> LLMResponseCache:
    """Get the global LLM response cache."""
    global _llm_cache
    if _llm_cache is None:
        _llm_cache = LLMResponseCache(CacheConfig.from_env())
    return _llm_cache
