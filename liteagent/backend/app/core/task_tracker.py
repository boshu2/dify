"""
Task lifecycle tracking for LiteAgent.

Provides:
- Task status management
- Task ownership verification
- Stop flag handling for graceful termination
- Automatic cleanup of orphaned tasks
"""
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from app.core.redis_client import redis_client

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


# Singleton instance
task_tracker = TaskTracker()
