"""
Production session and task management for LiteAgent.

Provides:
- Conversation session management with Redis persistence
- Task lifecycle tracking with ownership
- Distributed locks for resource protection
- Stop flag handling for graceful termination
"""
import json
import logging
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Generator

from app.core.config import get_settings
from app.core.redis_client import redis_client, redis_fallback

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    """Task execution status."""

    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"


@dataclass
class ConversationSession:
    """
    Conversation session state.

    Persisted in Redis for cross-request access.
    """

    session_id: str
    tenant_id: str
    agent_id: str
    user_id: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "session_id": self.session_id,
            "tenant_id": self.tenant_id,
            "agent_id": self.agent_id,
            "user_id": self.user_id,
            "messages": self.messages,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConversationSession":
        """Deserialize from dictionary."""
        return cls(
            session_id=data["session_id"],
            tenant_id=data["tenant_id"],
            agent_id=data["agent_id"],
            user_id=data["user_id"],
            messages=data.get("messages", []),
            metadata=data.get("metadata", {}),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
        )


class SessionManager:
    """
    Manages conversation sessions in Redis.

    Features:
    - Session creation and retrieval
    - Message history management
    - Automatic TTL-based expiration
    """

    KEY_PREFIX = "liteagent:session"

    def __init__(self, ttl: int | None = None):
        settings = get_settings()
        self.ttl = ttl or settings.cache_conversation_ttl

    def _get_key(self, session_id: str) -> str:
        """Generate Redis key for session."""
        return f"{self.KEY_PREFIX}:{session_id}"

    def _get_index_key(self, tenant_id: str, agent_id: str) -> str:
        """Generate Redis key for session index."""
        return f"{self.KEY_PREFIX}:index:{tenant_id}:{agent_id}"

    def create(
        self,
        tenant_id: str,
        agent_id: str,
        user_id: str,
        metadata: dict[str, Any] = None,
    ) -> ConversationSession:
        """
        Create a new conversation session.

        Args:
            tenant_id: Tenant identifier
            agent_id: Agent identifier
            user_id: User identifier
            metadata: Optional session metadata

        Returns:
            New ConversationSession
        """
        session = ConversationSession(
            session_id=str(uuid.uuid4()),
            tenant_id=tenant_id,
            agent_id=agent_id,
            user_id=user_id,
            metadata=metadata or {},
        )

        self._save(session)

        # Add to index for listing
        index_key = self._get_index_key(tenant_id, agent_id)
        try:
            redis_client.zadd(
                index_key,
                {session.session_id: time.time()},
            )
            redis_client.expire(index_key, self.ttl * 10)  # Keep index longer
        except Exception as e:
            logger.warning(f"Failed to update session index: {e}")

        return session

    def get(self, session_id: str) -> ConversationSession | None:
        """
        Get session by ID.

        Returns:
            ConversationSession or None if not found
        """
        key = self._get_key(session_id)

        try:
            data = redis_client.get(key)
            if data is None:
                return None

            session = ConversationSession.from_dict(
                json.loads(data.decode("utf-8"))
            )

            # Refresh TTL on access
            redis_client.expire(key, self.ttl)

            return session

        except Exception as e:
            logger.warning(f"Failed to get session: {e}")
            return None

    def _save(self, session: ConversationSession) -> bool:
        """Save session to Redis."""
        key = self._get_key(session.session_id)

        try:
            session.updated_at = datetime.now(timezone.utc).isoformat()
            redis_client.setex(
                key,
                self.ttl,
                json.dumps(session.to_dict()),
            )
            return True
        except Exception as e:
            logger.warning(f"Failed to save session: {e}")
            return False

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: dict[str, Any] = None,
    ) -> bool:
        """
        Add message to session.

        Args:
            session_id: Session ID
            role: Message role (user, assistant, system)
            content: Message content
            metadata: Optional message metadata

        Returns:
            True if successful
        """
        session = self.get(session_id)
        if session is None:
            return False

        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata or {},
        }

        session.messages.append(message)

        # Trim to reasonable size (keep last 100 messages)
        if len(session.messages) > 100:
            session.messages = session.messages[-100:]

        return self._save(session)

    def get_messages(
        self,
        session_id: str,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Get recent messages from session."""
        session = self.get(session_id)
        if session is None:
            return []

        return session.messages[-limit:]

    def list_sessions(
        self,
        tenant_id: str,
        agent_id: str,
        limit: int = 20,
    ) -> list[str]:
        """List recent session IDs for tenant/agent."""
        index_key = self._get_index_key(tenant_id, agent_id)

        try:
            # Get most recent sessions
            session_ids = redis_client.zrevrange(index_key, 0, limit - 1)
            return [sid.decode("utf-8") for sid in session_ids]
        except Exception as e:
            logger.warning(f"Failed to list sessions: {e}")
            return []

    def delete(self, session_id: str) -> bool:
        """Delete session."""
        session = self.get(session_id)
        if session is None:
            return False

        key = self._get_key(session_id)
        index_key = self._get_index_key(session.tenant_id, session.agent_id)

        try:
            redis_client.delete(key)
            redis_client.zrem(index_key, session_id)
            return True
        except Exception as e:
            logger.warning(f"Failed to delete session: {e}")
            return False


@dataclass
class TaskInfo:
    """Task execution information."""

    task_id: str
    agent_id: str
    user_id: str
    status: TaskStatus
    started_at: str
    updated_at: str
    metadata: dict[str, Any] = field(default_factory=dict)


class TaskTracker:
    """
    Tracks task lifecycle and ownership.

    Features:
    - Task ownership verification
    - Stop flag handling
    - Automatic cleanup of orphaned tasks
    """

    KEY_PREFIX = "liteagent:task"
    TASK_TTL = 1800  # 30 minutes
    STOP_FLAG_TTL = 600  # 10 minutes

    def __init__(self):
        pass

    def _get_task_key(self, task_id: str) -> str:
        """Generate Redis key for task."""
        return f"{self.KEY_PREFIX}:{task_id}"

    def _get_stop_key(self, task_id: str) -> str:
        """Generate Redis key for stop flag."""
        return f"{self.KEY_PREFIX}:stop:{task_id}"

    def _get_ownership_key(self, task_id: str) -> str:
        """Generate Redis key for task ownership."""
        return f"{self.KEY_PREFIX}:owner:{task_id}"

    def start_task(
        self,
        task_id: str,
        agent_id: str,
        user_id: str,
        metadata: dict[str, Any] = None,
    ) -> TaskInfo:
        """
        Register task start.

        Args:
            task_id: Unique task identifier
            agent_id: Agent executing the task
            user_id: User who initiated the task
            metadata: Optional task metadata

        Returns:
            TaskInfo with task details
        """
        now = datetime.now(timezone.utc).isoformat()

        info = TaskInfo(
            task_id=task_id,
            agent_id=agent_id,
            user_id=user_id,
            status=TaskStatus.RUNNING,
            started_at=now,
            updated_at=now,
            metadata=metadata or {},
        )

        task_key = self._get_task_key(task_id)
        ownership_key = self._get_ownership_key(task_id)

        try:
            with redis_client.pipeline() as pipe:
                # Store task info
                pipe.setex(
                    task_key,
                    self.TASK_TTL,
                    json.dumps({
                        "task_id": info.task_id,
                        "agent_id": info.agent_id,
                        "user_id": info.user_id,
                        "status": info.status.value,
                        "started_at": info.started_at,
                        "updated_at": info.updated_at,
                        "metadata": info.metadata,
                    }),
                )
                # Store ownership
                pipe.setex(ownership_key, self.TASK_TTL, user_id)
                pipe.execute()
        except Exception as e:
            logger.warning(f"Failed to start task tracking: {e}")

        return info

    def get_task(self, task_id: str) -> TaskInfo | None:
        """Get task info by ID."""
        key = self._get_task_key(task_id)

        try:
            data = redis_client.get(key)
            if data is None:
                return None

            parsed = json.loads(data.decode("utf-8"))
            return TaskInfo(
                task_id=parsed["task_id"],
                agent_id=parsed["agent_id"],
                user_id=parsed["user_id"],
                status=TaskStatus(parsed["status"]),
                started_at=parsed["started_at"],
                updated_at=parsed["updated_at"],
                metadata=parsed.get("metadata", {}),
            )
        except Exception as e:
            logger.warning(f"Failed to get task: {e}")
            return None

    def update_status(
        self,
        task_id: str,
        status: TaskStatus,
        metadata: dict[str, Any] = None,
    ) -> bool:
        """Update task status."""
        info = self.get_task(task_id)
        if info is None:
            return False

        info.status = status
        info.updated_at = datetime.now(timezone.utc).isoformat()
        if metadata:
            info.metadata.update(metadata)

        key = self._get_task_key(task_id)

        try:
            redis_client.setex(
                key,
                self.TASK_TTL,
                json.dumps({
                    "task_id": info.task_id,
                    "agent_id": info.agent_id,
                    "user_id": info.user_id,
                    "status": info.status.value,
                    "started_at": info.started_at,
                    "updated_at": info.updated_at,
                    "metadata": info.metadata,
                }),
            )
            return True
        except Exception as e:
            logger.warning(f"Failed to update task status: {e}")
            return False

    def complete_task(
        self,
        task_id: str,
        result: Any = None,
    ) -> bool:
        """Mark task as completed."""
        return self.update_status(
            task_id,
            TaskStatus.COMPLETED,
            {"result": result} if result else None,
        )

    def fail_task(
        self,
        task_id: str,
        error: str,
    ) -> bool:
        """Mark task as failed."""
        return self.update_status(
            task_id,
            TaskStatus.FAILED,
            {"error": error},
        )

    def set_stop_flag(
        self,
        task_id: str,
        user_id: str,
    ) -> bool:
        """
        Set stop flag for task (only if user owns it).

        Args:
            task_id: Task to stop
            user_id: User requesting stop

        Returns:
            True if stop flag was set
        """
        ownership_key = self._get_ownership_key(task_id)
        stop_key = self._get_stop_key(task_id)

        try:
            # Verify ownership
            owner = redis_client.get(ownership_key)
            if owner is None:
                return False  # Task not found

            if owner.decode("utf-8") != user_id:
                logger.warning(
                    f"Unauthorized stop attempt: task={task_id}, "
                    f"owner={owner.decode('utf-8')}, requester={user_id}"
                )
                return False

            # Set stop flag
            redis_client.setex(stop_key, self.STOP_FLAG_TTL, "1")
            return True

        except Exception as e:
            logger.warning(f"Failed to set stop flag: {e}")
            return False

    def is_stopped(self, task_id: str) -> bool:
        """Check if stop flag is set for task."""
        stop_key = self._get_stop_key(task_id)

        try:
            return bool(redis_client.exists(stop_key))
        except Exception:
            return False

    def clear_task(self, task_id: str) -> None:
        """Clear all task-related keys."""
        keys = [
            self._get_task_key(task_id),
            self._get_stop_key(task_id),
            self._get_ownership_key(task_id),
        ]

        try:
            redis_client.delete(*keys)
        except Exception as e:
            logger.warning(f"Failed to clear task: {e}")


class DistributedLock:
    """
    Redis-based distributed lock.

    Features:
    - Automatic expiration (prevents deadlocks)
    - Owner tracking (only owner can release)
    - Context manager support
    """

    KEY_PREFIX = "liteagent:lock"

    def __init__(
        self,
        name: str,
        timeout: int = 60,
        blocking: bool = True,
        blocking_timeout: float = 10.0,
    ):
        self.name = name
        self.timeout = timeout
        self.blocking = blocking
        self.blocking_timeout = blocking_timeout
        self._key = f"{self.KEY_PREFIX}:{name}"
        self._owner_id = str(uuid.uuid4())
        self._acquired = False

    def acquire(self) -> bool:
        """
        Acquire the lock.

        Returns:
            True if lock acquired
        """
        start_time = time.time()

        while True:
            try:
                # Try to set lock with NX (only if not exists)
                result = redis_client.set(
                    self._key,
                    self._owner_id,
                    nx=True,
                    ex=self.timeout,
                )

                if result:
                    self._acquired = True
                    return True

                if not self.blocking:
                    return False

                # Check timeout
                elapsed = time.time() - start_time
                if elapsed >= self.blocking_timeout:
                    return False

                # Wait before retry
                time.sleep(0.1)

            except Exception as e:
                logger.warning(f"Lock acquire failed: {e}")
                return False

        return False

    def release(self) -> bool:
        """
        Release the lock.

        Only releases if we own the lock.
        """
        if not self._acquired:
            return False

        try:
            # Check ownership before releasing
            current_owner = redis_client.get(self._key)
            if current_owner and current_owner.decode("utf-8") == self._owner_id:
                redis_client.delete(self._key)
                self._acquired = False
                return True
            return False
        except Exception as e:
            logger.warning(f"Lock release failed: {e}")
            return False

    def extend(self, additional_time: int | None = None) -> bool:
        """Extend lock timeout."""
        if not self._acquired:
            return False

        try:
            # Verify we still own the lock
            current_owner = redis_client.get(self._key)
            if current_owner and current_owner.decode("utf-8") == self._owner_id:
                redis_client.expire(self._key, additional_time or self.timeout)
                return True
            return False
        except Exception:
            return False

    def __enter__(self) -> "DistributedLock":
        if not self.acquire():
            raise TimeoutError(f"Failed to acquire lock: {self.name}")
        return self

    def __exit__(self, *args) -> None:
        self.release()


@contextmanager
def distributed_lock(
    name: str,
    timeout: int = 60,
    blocking: bool = True,
    blocking_timeout: float = 10.0,
) -> Generator[DistributedLock, None, None]:
    """
    Context manager for distributed locks.

    Usage:
        with distributed_lock("my-resource") as lock:
            # Critical section
            pass
    """
    lock = DistributedLock(name, timeout, blocking, blocking_timeout)

    if not lock.acquire():
        raise TimeoutError(f"Failed to acquire lock: {name}")

    try:
        yield lock
    finally:
        lock.release()


# Singleton instances
session_manager = SessionManager()
task_tracker = TaskTracker()
