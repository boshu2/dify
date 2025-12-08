"""
Tests for Redis infrastructure components.

Tests caching, rate limiting, pub/sub, and session management.
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
from app.core.rate_limiter import (
    SlidingWindowRateLimiter,
    ConcurrentRequestLimiter,
    CombinedRateLimiter,
    RateLimitExceededError,
    ConcurrentLimitExceededError,
    RateLimitInfo,
)
from app.core.pubsub import (
    EventType,
    AgentEvent,
    Topic,
    AgentEventPublisher,
    CommandChannel,
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
# Rate Limiter Tests
# =============================================================================


class TestRateLimitInfo:
    """Tests for RateLimitInfo dataclass."""

    def test_create_info(self):
        """Should create rate limit info."""
        info = RateLimitInfo(
            limit=60,
            remaining=55,
            reset_at=int(time.time()) + 60,
            concurrent_limit=10,
            concurrent_active=3,
        )

        assert info.limit == 60
        assert info.remaining == 55
        assert info.concurrent_limit == 10
        assert info.concurrent_active == 3


class TestRateLimitExceededError:
    """Tests for rate limit errors."""

    def test_exceeded_error(self):
        """Should create proper error response."""
        error = RateLimitExceededError(limit=60, window=30)

        assert error.limit == 60
        assert error.window == 30
        assert "60" in error.message

        response = error.to_response()
        assert response["status_code"] == 429
        assert "Retry-After" in response["headers"]

    def test_concurrent_error(self):
        """Should create concurrent limit error."""
        error = ConcurrentLimitExceededError(max_concurrent=10)

        assert error.max_concurrent == 10
        assert "10" in error.message


class TestSlidingWindowRateLimiter:
    """Tests for sliding window rate limiter."""

    def test_init_from_settings(self):
        """Should initialize with settings."""
        with patch("app.core.rate_limiter.get_settings") as mock_settings:
            mock_settings.return_value.rate_limit_requests_per_minute = 100
            limiter = SlidingWindowRateLimiter()
            assert limiter.requests_per_minute == 100

    def test_custom_limit(self):
        """Should accept custom limit."""
        limiter = SlidingWindowRateLimiter(requests_per_minute=30)
        assert limiter.requests_per_minute == 30

    def test_key_generation(self):
        """Should generate correct keys."""
        limiter = SlidingWindowRateLimiter()
        key = limiter._get_key("client_123")

        assert "liteagent" in key
        assert "rate_limit" in key
        assert "sliding" in key
        assert "client_123" in key


class TestConcurrentRequestLimiter:
    """Tests for concurrent request limiter."""

    def test_init_from_settings(self):
        """Should initialize with settings."""
        with patch("app.core.rate_limiter.get_settings") as mock_settings:
            mock_settings.return_value.rate_limit_max_concurrent = 20
            limiter = ConcurrentRequestLimiter()
            assert limiter.max_concurrent == 20

    def test_unlimited_when_disabled(self):
        """Should return unlimited ID when max is 0."""
        limiter = ConcurrentRequestLimiter(max_concurrent=0)
        request_id = limiter.enter("client")
        assert request_id == "unlimited"


class TestCombinedRateLimiter:
    """Tests for combined rate limiter."""

    def test_has_both_limiters(self):
        """Should have both sliding window and concurrent limiters."""
        limiter = CombinedRateLimiter(
            requests_per_minute=60,
            max_concurrent=10,
        )

        assert limiter.sliding_window is not None
        assert limiter.concurrent is not None
        assert limiter.sliding_window.requests_per_minute == 60
        assert limiter.concurrent.max_concurrent == 10


# =============================================================================
# Pub/Sub Tests
# =============================================================================


class TestAgentEvent:
    """Tests for AgentEvent."""

    def test_create_event(self):
        """Should create agent event."""
        event = AgentEvent(
            event_type=EventType.AGENT_STARTED,
            agent_id="agent_123",
            execution_id="exec_456",
            data={"input": "Hello"},
        )

        assert event.event_type == EventType.AGENT_STARTED
        assert event.agent_id == "agent_123"
        assert event.execution_id == "exec_456"
        assert event.data["input"] == "Hello"
        assert event.timestamp  # Should have timestamp

    def test_serialize_deserialize(self):
        """Should serialize and deserialize events."""
        event = AgentEvent(
            event_type=EventType.STREAM_CHUNK,
            agent_id="agent_123",
            execution_id="exec_456",
            data={"content": "Hello"},
        )

        json_str = event.to_json()
        assert isinstance(json_str, str)

        restored = AgentEvent.from_json(json_str)
        assert restored.event_type == event.event_type
        assert restored.agent_id == event.agent_id
        assert restored.data == event.data

    def test_deserialize_from_bytes(self):
        """Should deserialize from bytes."""
        event = AgentEvent(
            event_type=EventType.AGENT_COMPLETED,
            agent_id="agent_123",
            execution_id="exec_456",
        )

        json_bytes = event.to_json().encode("utf-8")
        restored = AgentEvent.from_json(json_bytes)

        assert restored.event_type == event.event_type


class TestEventType:
    """Tests for EventType enum."""

    def test_all_event_types_exist(self):
        """Should have all required event types."""
        assert EventType.AGENT_STARTED
        assert EventType.AGENT_STEP
        assert EventType.AGENT_COMPLETED
        assert EventType.AGENT_FAILED
        assert EventType.STREAM_CHUNK
        assert EventType.STREAM_TOOL_CALL
        assert EventType.CONTROL_PAUSE
        assert EventType.HUMAN_INPUT_REQUESTED


class TestTopic:
    """Tests for Topic."""

    def test_topic_key(self):
        """Should generate correct topic key."""
        topic = Topic("test-topic")
        assert "liteagent:pubsub:test-topic" in topic._key


class TestAgentEventPublisher:
    """Tests for AgentEventPublisher."""

    def test_publisher_topics(self):
        """Should create correct topics."""
        publisher = AgentEventPublisher("agent_123", "exec_456")

        assert publisher.agent_id == "agent_123"
        assert publisher.execution_id == "exec_456"
        assert publisher._topic is not None
        assert publisher._global_topic is not None


class TestCommandChannel:
    """Tests for CommandChannel."""

    def test_channel_key(self):
        """Should generate correct channel key."""
        channel = CommandChannel("agent_123", "exec_456")

        assert "liteagent:commands" in channel._key
        assert "agent_123" in channel._key
        assert "exec_456" in channel._key


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


class TestRateLimiterWithMockedRedis:
    """Tests for rate limiter with mocked Redis."""

    def test_sliding_window_allows_within_limit(self):
        """Should allow requests within limit."""
        mock_client = MagicMock()
        mock_pipeline = MagicMock()
        mock_pipeline.__enter__ = MagicMock(return_value=mock_pipeline)
        mock_pipeline.__exit__ = MagicMock(return_value=False)
        mock_pipeline.execute.return_value = [None, 5, None, None]  # 5 requests in window
        mock_client.pipeline.return_value = mock_pipeline

        with patch.object(
            __import__("app.core.rate_limiter", fromlist=["redis_client"]),
            "redis_client",
            mock_client,
        ):
            limiter = SlidingWindowRateLimiter(requests_per_minute=60)
            info = limiter.check("client_123")

            assert info.limit == 60
            assert info.remaining > 0

    def test_concurrent_limiter_tracks_requests(self):
        """Should track concurrent requests."""
        mock_client = MagicMock()
        mock_client.hlen.return_value = 3

        with patch.object(
            __import__("app.core.rate_limiter", fromlist=["redis_client"]),
            "redis_client",
            mock_client,
        ):
            limiter = ConcurrentRequestLimiter(max_concurrent=10)
            count = limiter.get_active_count("client_123")
            assert count == 3


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
            __import__("app.core.session_manager", fromlist=["redis_client"]),
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
            __import__("app.core.session_manager", fromlist=["redis_client"]),
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
            __import__("app.core.session_manager", fromlist=["redis_client"]),
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
            __import__("app.core.session_manager", fromlist=["redis_client"]),
            "redis_client",
            mock_client,
        ):
            lock = DistributedLock("test-resource", timeout=1, blocking=True, blocking_timeout=0.5)

            with pytest.raises(TimeoutError):
                with lock:
                    pass
