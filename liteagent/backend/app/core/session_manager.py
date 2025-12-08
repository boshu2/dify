"""
Conversation session management for LiteAgent.

Provides:
- Session creation and retrieval
- Message history management
- Automatic TTL-based expiration
"""
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from app.core.config import get_settings
from app.core.redis_client import redis_client

logger = logging.getLogger(__name__)


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


# Singleton instance
session_manager = SessionManager()


# Re-export for backward compatibility
from app.core.task_tracker import (  # noqa: E402, F401
    TaskStatus,
    TaskInfo,
    TaskTracker,
    task_tracker,
)
from app.core.distributed_lock import (  # noqa: E402, F401
    DistributedLock,
    distributed_lock,
)
