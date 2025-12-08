"""
Production caching layer for LiteAgent.

Provides type-safe caching for:
- Provider credentials (24h TTL)
- Embedding vectors (10 min TTL)
- Conversation context (1h TTL)
- Agent configurations (24h TTL)
"""
import base64
import hashlib
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Generic, TypeVar

import numpy as np

from app.core.config import get_settings
from app.core.redis_client import redis_client, redis_fallback

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class CacheEntry(Generic[T]):
    """Cached value with metadata."""

    value: T
    created_at: datetime
    ttl: int
    hits: int = 0


class BaseCache(ABC, Generic[T]):
    """
    Abstract base cache with common operations.

    Subclasses define serialization and key generation.
    """

    def __init__(self, prefix: str, default_ttl: int):
        self.prefix = prefix
        self.default_ttl = default_ttl

    @abstractmethod
    def _serialize(self, value: T) -> bytes:
        """Serialize value for storage."""
        ...

    @abstractmethod
    def _deserialize(self, data: bytes) -> T:
        """Deserialize value from storage."""
        ...

    def _make_key(self, *parts: str) -> str:
        """Generate cache key from parts."""
        return f"{self.prefix}:{':'.join(parts)}"

    def get(self, key: str) -> T | None:
        """Get cached value or None."""
        try:
            data = redis_client.get(key)
            if data is None:
                return None
            return self._deserialize(data)
        except Exception as e:
            logger.warning(f"Cache get failed: {e}")
            return None

    def set(self, key: str, value: T, ttl: int | None = None) -> bool:
        """Set cached value with TTL."""
        try:
            data = self._serialize(value)
            redis_client.setex(key, ttl or self.default_ttl, data)
            return True
        except Exception as e:
            logger.warning(f"Cache set failed: {e}")
            return False

    def delete(self, key: str) -> bool:
        """Delete cached value."""
        try:
            redis_client.delete(key)
            return True
        except Exception as e:
            logger.warning(f"Cache delete failed: {e}")
            return False

    def exists(self, key: str) -> bool:
        """Check if key exists."""
        try:
            return bool(redis_client.exists(key))
        except Exception:
            return False

    def touch(self, key: str, ttl: int | None = None) -> bool:
        """Reset TTL on existing key."""
        try:
            return bool(redis_client.expire(key, ttl or self.default_ttl))
        except Exception:
            return False


class CredentialsCache(BaseCache[dict[str, Any]]):
    """
    Cache for provider credentials.

    Multi-tenant safe with tenant_id in key.
    24-hour TTL for security balance.
    """

    def __init__(self):
        settings = get_settings()
        super().__init__("credentials", settings.cache_credentials_ttl)

    def _serialize(self, value: dict[str, Any]) -> bytes:
        return json.dumps(value).encode("utf-8")

    def _deserialize(self, data: bytes) -> dict[str, Any]:
        return json.loads(data.decode("utf-8"))

    def get_provider_credentials(
        self,
        tenant_id: str,
        provider_id: str,
    ) -> dict[str, Any] | None:
        """Get cached provider credentials."""
        key = self._make_key("provider", tenant_id, provider_id)
        return self.get(key)

    def set_provider_credentials(
        self,
        tenant_id: str,
        provider_id: str,
        credentials: dict[str, Any],
    ) -> bool:
        """Cache provider credentials."""
        key = self._make_key("provider", tenant_id, provider_id)
        return self.set(key, credentials)

    def invalidate_provider(self, tenant_id: str, provider_id: str) -> bool:
        """Invalidate cached credentials on update."""
        key = self._make_key("provider", tenant_id, provider_id)
        return self.delete(key)


class EmbeddingCache(BaseCache[list[float]]):
    """
    Cache for embedding vectors.

    Short TTL (10 min) for freshness.
    Uses content hash for deduplication.
    Binary encoding for efficiency.
    """

    def __init__(self):
        settings = get_settings()
        super().__init__("embedding", settings.cache_embeddings_ttl)

    def _serialize(self, value: list[float]) -> bytes:
        """Encode as base64 numpy array for compact storage."""
        arr = np.array(value, dtype=np.float32)
        return base64.b64encode(arr.tobytes())

    def _deserialize(self, data: bytes) -> list[float]:
        """Decode from base64 numpy array."""
        arr = np.frombuffer(base64.b64decode(data), dtype=np.float32)
        return arr.tolist()

    @staticmethod
    def _hash_text(text: str) -> str:
        """Generate consistent hash for text content."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    def get_embedding(
        self,
        provider: str,
        model: str,
        text: str,
    ) -> list[float] | None:
        """Get cached embedding for text."""
        text_hash = self._hash_text(text)
        key = self._make_key(provider, model, text_hash)

        result = self.get(key)
        if result is not None:
            # Extend TTL on hit (LRU-like behavior)
            self.touch(key)
        return result

    def set_embedding(
        self,
        provider: str,
        model: str,
        text: str,
        embedding: list[float],
    ) -> bool:
        """Cache embedding for text."""
        text_hash = self._hash_text(text)
        key = self._make_key(provider, model, text_hash)
        return self.set(key, embedding)


class ConversationCache(BaseCache[dict[str, Any]]):
    """
    Cache for conversation context.

    Stores recent messages and metadata for fast retrieval.
    1-hour TTL with refresh on activity.
    """

    def __init__(self):
        settings = get_settings()
        super().__init__("conversation", settings.cache_conversation_ttl)

    def _serialize(self, value: dict[str, Any]) -> bytes:
        return json.dumps(value, default=str).encode("utf-8")

    def _deserialize(self, data: bytes) -> dict[str, Any]:
        return json.loads(data.decode("utf-8"))

    def get_context(
        self,
        conversation_id: str,
    ) -> dict[str, Any] | None:
        """Get conversation context."""
        key = self._make_key(conversation_id)
        result = self.get(key)
        if result is not None:
            self.touch(key)
        return result

    def set_context(
        self,
        conversation_id: str,
        context: dict[str, Any],
    ) -> bool:
        """Cache conversation context."""
        key = self._make_key(conversation_id)
        return self.set(key, context)

    def append_message(
        self,
        conversation_id: str,
        message: dict[str, Any],
        max_messages: int = 50,
    ) -> bool:
        """Append message to conversation history."""
        key = self._make_key(conversation_id)

        context = self.get(key) or {"messages": [], "metadata": {}}
        context["messages"].append(message)

        # Trim to max messages (keep most recent)
        if len(context["messages"]) > max_messages:
            context["messages"] = context["messages"][-max_messages:]

        context["metadata"]["last_updated"] = datetime.now(timezone.utc).isoformat()

        return self.set(key, context)


class AgentConfigCache(BaseCache[dict[str, Any]]):
    """
    Cache for agent configurations.

    24-hour TTL, invalidated on config changes.
    """

    def __init__(self):
        settings = get_settings()
        super().__init__("agent_config", settings.cache_credentials_ttl)

    def _serialize(self, value: dict[str, Any]) -> bytes:
        return json.dumps(value, default=str).encode("utf-8")

    def _deserialize(self, data: bytes) -> dict[str, Any]:
        return json.loads(data.decode("utf-8"))

    def get_config(
        self,
        tenant_id: str,
        agent_id: str,
    ) -> dict[str, Any] | None:
        """Get cached agent configuration."""
        key = self._make_key(tenant_id, agent_id)
        return self.get(key)

    def set_config(
        self,
        tenant_id: str,
        agent_id: str,
        config: dict[str, Any],
    ) -> bool:
        """Cache agent configuration."""
        key = self._make_key(tenant_id, agent_id)
        return self.set(key, config)

    def invalidate(self, tenant_id: str, agent_id: str) -> bool:
        """Invalidate cached config on update."""
        key = self._make_key(tenant_id, agent_id)
        return self.delete(key)


# Singleton instances
credentials_cache = CredentialsCache()
embedding_cache = EmbeddingCache()
conversation_cache = ConversationCache()
agent_config_cache = AgentConfigCache()


@redis_fallback(default_return=None)
def get_cached_or_compute(
    cache_key: str,
    compute_fn: callable,
    ttl: int = 3600,
) -> Any:
    """
    Get from cache or compute and cache result.

    Common pattern for expensive operations.
    """
    cached = redis_client.get(cache_key)
    if cached is not None:
        return json.loads(cached.decode("utf-8"))

    result = compute_fn()

    if result is not None:
        redis_client.setex(
            cache_key,
            ttl,
            json.dumps(result, default=str).encode("utf-8"),
        )

    return result
