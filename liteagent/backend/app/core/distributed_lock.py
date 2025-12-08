"""
Distributed locking for LiteAgent.

Provides Redis-based distributed locks for:
- Resource protection across workers
- Preventing race conditions
- Coordinating access to shared resources
"""
import logging
import time
import uuid
from contextlib import contextmanager
from typing import Generator

from app.core.redis_client import redis_client

logger = logging.getLogger(__name__)


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
