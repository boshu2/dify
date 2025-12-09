"""
Unit tests for caching layer.
Tests in-memory and Redis-compatible caching for LLM responses.
"""
import pytest
import asyncio
import json
from datetime import timedelta
from unittest.mock import Mock, AsyncMock, patch

from app.core.cache import (
    CacheConfig,
    CacheKey,
    InMemoryCache,
    CacheEntry,
    LLMResponseCache,
    cache_key_from_messages,
)


class TestCacheConfig:
    """Tests for cache configuration."""

    def test_default_config(self):
        """Test default cache configuration."""
        config = CacheConfig()
        assert config.default_ttl_seconds == 3600
        assert config.max_size == 1000
        assert config.enable_compression is False

    def test_config_from_env(self):
        """Test configuration from environment variables."""
        with patch.dict(
            "os.environ",
            {
                "CACHE_DEFAULT_TTL": "7200",
                "CACHE_MAX_SIZE": "5000",
            },
        ):
            config = CacheConfig.from_env()
            assert config.default_ttl_seconds == 7200
            assert config.max_size == 5000


class TestCacheKey:
    """Tests for cache key generation."""

    def test_create_cache_key(self):
        """Test creating a cache key."""
        key = CacheKey(
            prefix="llm",
            provider="openai",
            model="gpt-4o",
            content_hash="abc123",
        )
        assert "llm" in str(key)
        assert "openai" in str(key)

    def test_cache_key_is_deterministic(self):
        """Test cache key is deterministic for same inputs."""
        key1 = CacheKey(prefix="llm", provider="openai", model="gpt-4o", content_hash="abc")
        key2 = CacheKey(prefix="llm", provider="openai", model="gpt-4o", content_hash="abc")
        assert str(key1) == str(key2)

    def test_different_inputs_different_keys(self):
        """Test different inputs produce different keys."""
        key1 = CacheKey(prefix="llm", provider="openai", model="gpt-4o", content_hash="abc")
        key2 = CacheKey(prefix="llm", provider="openai", model="gpt-4o", content_hash="xyz")
        assert str(key1) != str(key2)


class TestCacheKeyFromMessages:
    """Tests for generating cache keys from messages."""

    def test_cache_key_from_messages(self):
        """Test generating cache key from messages."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello!"},
        ]
        key = cache_key_from_messages("openai", "gpt-4o", messages)
        assert key is not None
        assert len(key) > 0

    def test_same_messages_same_key(self):
        """Test same messages produce same key."""
        messages = [{"role": "user", "content": "Hello!"}]
        key1 = cache_key_from_messages("openai", "gpt-4o", messages)
        key2 = cache_key_from_messages("openai", "gpt-4o", messages)
        assert key1 == key2

    def test_different_messages_different_keys(self):
        """Test different messages produce different keys."""
        key1 = cache_key_from_messages("openai", "gpt-4o", [{"role": "user", "content": "Hi"}])
        key2 = cache_key_from_messages("openai", "gpt-4o", [{"role": "user", "content": "Hello"}])
        assert key1 != key2


class TestCacheEntry:
    """Tests for cache entry."""

    def test_create_entry(self):
        """Test creating a cache entry."""
        entry = CacheEntry(
            value="cached response",
            created_at=1000.0,
            ttl_seconds=3600,
        )
        assert entry.value == "cached response"
        assert entry.ttl_seconds == 3600

    def test_entry_not_expired(self):
        """Test entry that hasn't expired."""
        import time
        entry = CacheEntry(
            value="data",
            created_at=time.monotonic(),
            ttl_seconds=3600,
        )
        assert entry.is_expired is False

    def test_entry_expired(self):
        """Test entry that has expired."""
        import time
        entry = CacheEntry(
            value="data",
            created_at=time.monotonic() - 7200,  # 2 hours ago
            ttl_seconds=3600,  # 1 hour TTL
        )
        assert entry.is_expired is True


class TestInMemoryCache:
    """Tests for in-memory cache."""

    @pytest.fixture
    def cache(self):
        """Create an in-memory cache."""
        config = CacheConfig(default_ttl_seconds=60, max_size=10)
        return InMemoryCache(config)

    @pytest.mark.asyncio
    async def test_set_and_get(self, cache):
        """Test setting and getting a value."""
        await cache.set("key1", "value1")
        result = await cache.get("key1")
        assert result == "value1"

    @pytest.mark.asyncio
    async def test_get_nonexistent(self, cache):
        """Test getting a nonexistent key."""
        result = await cache.get("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete(self, cache):
        """Test deleting a key."""
        await cache.set("key1", "value1")
        await cache.delete("key1")
        result = await cache.get("key1")
        assert result is None

    @pytest.mark.asyncio
    async def test_custom_ttl(self, cache):
        """Test custom TTL for entry."""
        await cache.set("key1", "value1", ttl_seconds=1)
        await asyncio.sleep(1.5)
        result = await cache.get("key1")
        assert result is None

    @pytest.mark.asyncio
    async def test_eviction_on_max_size(self, cache):
        """Test LRU eviction when max size reached."""
        # Fill cache to max
        for i in range(10):
            await cache.set(f"key{i}", f"value{i}")

        # Add one more, should evict oldest
        await cache.set("key_new", "value_new")

        # First key should be evicted
        result = await cache.get("key0")
        assert result is None

        # New key should exist
        result = await cache.get("key_new")
        assert result == "value_new"

    @pytest.mark.asyncio
    async def test_exists(self, cache):
        """Test checking if key exists."""
        await cache.set("key1", "value1")
        assert await cache.exists("key1") is True
        assert await cache.exists("nonexistent") is False

    @pytest.mark.asyncio
    async def test_clear(self, cache):
        """Test clearing the cache."""
        await cache.set("key1", "value1")
        await cache.set("key2", "value2")
        await cache.clear()
        assert await cache.get("key1") is None
        assert await cache.get("key2") is None

    @pytest.mark.asyncio
    async def test_get_stats(self, cache):
        """Test getting cache statistics."""
        await cache.set("key1", "value1")
        await cache.get("key1")  # Hit
        await cache.get("nonexistent")  # Miss

        stats = cache.get_stats()
        assert stats["hits"] >= 1
        assert stats["misses"] >= 1
        assert stats["size"] == 1


class TestLLMResponseCache:
    """Tests for LLM response caching."""

    @pytest.fixture
    def llm_cache(self):
        """Create an LLM response cache."""
        config = CacheConfig(default_ttl_seconds=3600)
        return LLMResponseCache(config)

    @pytest.mark.asyncio
    async def test_cache_response(self, llm_cache):
        """Test caching an LLM response."""
        messages = [{"role": "user", "content": "Hello!"}]
        response = {"content": "Hi there!", "usage": {"tokens": 10}}

        await llm_cache.cache_response("openai", "gpt-4o", messages, response)

        cached = await llm_cache.get_cached_response("openai", "gpt-4o", messages)
        assert cached is not None
        assert cached["content"] == "Hi there!"

    @pytest.mark.asyncio
    async def test_cache_miss(self, llm_cache):
        """Test cache miss for uncached messages."""
        messages = [{"role": "user", "content": "Unknown query"}]
        cached = await llm_cache.get_cached_response("openai", "gpt-4o", messages)
        assert cached is None

    @pytest.mark.asyncio
    async def test_different_models_different_cache(self, llm_cache):
        """Test different models have separate cache entries."""
        messages = [{"role": "user", "content": "Hello!"}]

        await llm_cache.cache_response("openai", "gpt-4o", messages, {"content": "gpt-4o response"})
        await llm_cache.cache_response("openai", "gpt-3.5-turbo", messages, {"content": "gpt-3.5 response"})

        cached_4o = await llm_cache.get_cached_response("openai", "gpt-4o", messages)
        cached_35 = await llm_cache.get_cached_response("openai", "gpt-3.5-turbo", messages)

        assert cached_4o["content"] == "gpt-4o response"
        assert cached_35["content"] == "gpt-3.5 response"
