"""
Tests for Redis infrastructure components.

Tests caching and session management.
Uses mock Redis for unit tests.
"""
import json
import time
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from app.core.cache_manager import (
    CredentialsCache,
    EmbeddingCache,
    ConversationCache,
    AgentConfigCache,
)
from app.core.session_manager import (
    ConversationSession,
    SessionManager,
    TaskTracker,
    TaskStatus,
    DistributedLock,
)


# =============================================================================
# Cache Manager Tests
# =============================================================================


class TestCredentialsCache:
    """Tests for credentials caching."""

    def test_serialize_deserialize(self):
        """Should serialize and deserialize credentials."""
        cache = CredentialsCache()
        credentials = {"api_key": "sk-test", "model": "gpt-4"}

        serialized = cache._serialize(credentials)
        assert isinstance(serialized, bytes)

        deserialized = cache._deserialize(serialized)
        assert deserialized == credentials

    def test_make_key(self):
        """Should generate correct cache key."""
        cache = CredentialsCache()
        key = cache._make_key("provider", "tenant_123", "provider_456")

        assert "credentials" in key
        assert "tenant_123" in key
        assert "provider_456" in key


class TestEmbeddingCache:
    """Tests for embedding caching."""

    def test_serialize_deserialize(self):
        """Should serialize and deserialize embeddings."""
        cache = EmbeddingCache()
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]

        serialized = cache._serialize(embedding)
        assert isinstance(serialized, bytes)

        deserialized = cache._deserialize(serialized)
        assert len(deserialized) == 5
        # Float precision may vary slightly
        assert abs(deserialized[0] - 0.1) < 0.001

    def test_hash_text(self):
        """Should generate consistent hashes."""
        cache = EmbeddingCache()

        hash1 = cache._hash_text("Hello world")
        hash2 = cache._hash_text("Hello world")
        hash3 = cache._hash_text("Different text")

        assert hash1 == hash2
        assert hash1 != hash3
        assert len(hash1) == 16


class TestConversationCache:
    """Tests for conversation caching."""

    def test_serialize_deserialize(self):
        """Should serialize and deserialize conversation context."""
        cache = ConversationCache()
        context = {
            "messages": [{"role": "user", "content": "Hello"}],
            "metadata": {"agent_id": "test"},
        }

        serialized = cache._serialize(context)
        assert isinstance(serialized, bytes)

        deserialized = cache._deserialize(serialized)
        assert deserialized == context


# =============================================================================
# Session Manager Tests
# =============================================================================


class TestConversationSession:
    """Tests for ConversationSession."""

    def test_create_session(self):
        """Should create session with defaults."""
        session = ConversationSession(
            session_id="sess_123",
            tenant_id="tenant_1",
            agent_id="agent_1",
            user_id="user_1",
        )

        assert session.session_id == "sess_123"
        assert session.messages == []
        assert session.metadata == {}
        assert session.created_at

    def test_serialize_deserialize(self):
        """Should serialize and deserialize session."""
        session = ConversationSession(
            session_id="sess_123",
            tenant_id="tenant_1",
            agent_id="agent_1",
            user_id="user_1",
            messages=[{"role": "user", "content": "Hello"}],
            metadata={"key": "value"},
        )

        data = session.to_dict()
        restored = ConversationSession.from_dict(data)

        assert restored.session_id == session.session_id
        assert restored.messages == session.messages
        assert restored.metadata == session.metadata


class TestSessionManager:
    """Tests for SessionManager."""

    def test_key_generation(self):
        """Should generate correct keys."""
        manager = SessionManager()

        session_key = manager._get_key("sess_123")
        assert "liteagent:session:sess_123" == session_key

        index_key = manager._get_index_key("tenant_1", "agent_1")
        assert "liteagent:session:index" in index_key
        assert "tenant_1" in index_key
        assert "agent_1" in index_key


class TestTaskTracker:
    """Tests for TaskTracker."""

    def test_key_generation(self):
        """Should generate correct keys."""
        tracker = TaskTracker()

        task_key = tracker._get_task_key("task_123")
        assert "liteagent:task:task_123" == task_key

        stop_key = tracker._get_stop_key("task_123")
        assert "stop" in stop_key

        owner_key = tracker._get_ownership_key("task_123")
        assert "owner" in owner_key


class TestTaskStatus:
    """Tests for TaskStatus enum."""

    def test_all_statuses_exist(self):
        """Should have all required statuses."""
        assert TaskStatus.PENDING
        assert TaskStatus.RUNNING
        assert TaskStatus.PAUSED
        assert TaskStatus.COMPLETED
        assert TaskStatus.FAILED
        assert TaskStatus.ABORTED


class TestDistributedLock:
    """Tests for DistributedLock."""

    def test_lock_key(self):
        """Should generate correct lock key."""
        lock = DistributedLock("my-resource")
        assert "liteagent:lock:my-resource" == lock._key

    def test_unique_owner_id(self):
        """Each lock should have unique owner."""
        lock1 = DistributedLock("resource")
        lock2 = DistributedLock("resource")

        assert lock1._owner_id != lock2._owner_id

    def test_not_acquired_initially(self):
        """Lock should not be acquired on creation."""
        lock = DistributedLock("resource")
        assert not lock._acquired


# =============================================================================
# Integration-Style Tests (with Mocked Redis)
# =============================================================================


class TestCacheWithMockedRedis:
    """Tests for cache operations with mocked Redis."""

    def test_credentials_get_set(self):
        """Should get and set credentials."""
        mock_client = MagicMock()

        with patch.object(
            __import__("app.core.cache_manager", fromlist=["redis_client"]),
            "redis_client",
            mock_client,
        ):
            cache = CredentialsCache()
            creds = {"api_key": "test-key"}

            # Mock get returning None (cache miss)
            mock_client.get.return_value = None
            result = cache.get("test-key")
            assert result is None

            # Mock get returning cached data
            mock_client.get.return_value = json.dumps(creds).encode()
            result = cache.get("test-key")
            assert result == creds


class TestSessionManagerWithMockedRedis:
    """Tests for session manager with mocked Redis."""

    def test_create_session(self):
        """Should create new session."""
        mock_client = MagicMock()

        with patch.object(
            __import__("app.core.session_manager", fromlist=["redis_client"]),
            "redis_client",
            mock_client,
        ):
            manager = SessionManager()
            session = manager.create(
                tenant_id="tenant_1",
                agent_id="agent_1",
                user_id="user_1",
            )

            assert session.session_id
            assert session.tenant_id == "tenant_1"
            mock_client.setex.assert_called()


class TestDistributedLockWithMockedRedis:
    """Tests for distributed lock with mocked Redis."""

    def test_acquire_success(self):
        """Should acquire lock when available."""
        mock_client = MagicMock()
        mock_client.set.return_value = True

        with patch.object(
            __import__("app.core.distributed_lock", fromlist=["redis_client"]),
            "redis_client",
            mock_client,
        ):
            lock = DistributedLock("test-resource")
            result = lock.acquire()

            assert result is True
            assert lock._acquired is True
            mock_client.set.assert_called_once()

    def test_acquire_failure(self):
        """Should fail to acquire when locked."""
        mock_client = MagicMock()
        mock_client.set.return_value = False

        with patch.object(
            __import__("app.core.distributed_lock", fromlist=["redis_client"]),
            "redis_client",
            mock_client,
        ):
            lock = DistributedLock("test-resource", blocking=False)
            result = lock.acquire()

            assert result is False
            assert lock._acquired is False

    def test_release_only_if_owner(self):
        """Should only release if owner."""
        mock_client = MagicMock()

        with patch.object(
            __import__("app.core.distributed_lock", fromlist=["redis_client"]),
            "redis_client",
            mock_client,
        ):
            lock = DistributedLock("test-resource")
            lock._acquired = True
            mock_client.get.return_value = lock._owner_id.encode()

            result = lock.release()

            assert result is True
            mock_client.delete.assert_called()

    def test_context_manager_timeout(self):
        """Should timeout when lock cannot be acquired."""
        mock_client = MagicMock()
        # set returns False to simulate lock already held
        mock_client.set.return_value = False

        with patch.object(
            __import__("app.core.distributed_lock", fromlist=["redis_client"]),
            "redis_client",
            mock_client,
        ):
            lock = DistributedLock("test-resource", timeout=1, blocking=True, blocking_timeout=0.5)

            with pytest.raises(TimeoutError):
                with lock:
                    pass
