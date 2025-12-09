"""
Production-grade rate limiting for LiteAgent.

Implements:
- Sliding window rate limiting (per-minute) using Redis sorted sets
- Concurrent request limiting using Redis hash maps
- Per-tenant and per-agent isolation
- Automatic cleanup of orphaned requests
"""
import logging
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Generator

from app.core.config import get_settings
from app.core.redis_client import redis_client, redis_fallback

logger = logging.getLogger(__name__)


class RateLimitError(Exception):
    """Base exception for rate limiting errors."""

    def __init__(self, message: str, retry_after: int = 60):
        self.message = message
        self.retry_after = retry_after
        super().__init__(message)

    def to_response(self) -> dict[str, Any]:
        """Convert to HTTP response details."""
        return {
            "status_code": 429,
            "detail": self.message,
            "headers": {
                "Retry-After": str(self.retry_after),
                "X-RateLimit-Reset": str(int(time.time()) + self.retry_after),
            },
        }


class RateLimitExceededError(RateLimitError):
    """Raised when request rate exceeds limit."""

    def __init__(self, limit: int, window: int = 60):
        super().__init__(
            f"Rate limit exceeded. Maximum {limit} requests per {window} seconds.",
            retry_after=window,
        )
        self.limit = limit
        self.window = window


class ConcurrentLimitExceededError(RateLimitError):
    """Raised when concurrent request limit is exceeded."""

    def __init__(self, max_concurrent: int):
        super().__init__(
            f"Too many concurrent requests. Maximum {max_concurrent} active requests.",
            retry_after=5,
        )
        self.max_concurrent = max_concurrent


@dataclass
class RateLimitInfo:
    """Current rate limit status."""

    limit: int
    remaining: int
    reset_at: int
    concurrent_limit: int
    concurrent_active: int


class SlidingWindowRateLimiter:
    """
    Sliding window rate limiter using Redis sorted sets.

    Advantages over fixed window:
    - No burst at window boundaries
    - Accurate request counting
    - Automatic cleanup with ZREMRANGEBYSCORE
    """

    KEY_PREFIX = "liteagent:rate_limit"

    def __init__(
        self,
        requests_per_minute: int | None = None,
        window_seconds: int = 60,
    ):
        settings = get_settings()
        self.requests_per_minute = (
            requests_per_minute if requests_per_minute is not None
            else settings.rate_limit_requests_per_minute
        )
        self.window_seconds = window_seconds
        self.window_ms = window_seconds * 1000

    def _get_key(self, client_id: str) -> str:
        """Generate Redis key for client."""
        return f"{self.KEY_PREFIX}:sliding:{client_id}"

    def check(self, client_id: str) -> RateLimitInfo:
        """
        Check and record a request.

        Args:
            client_id: Unique client identifier (tenant_id, api_key, etc.)

        Returns:
            RateLimitInfo with current status

        Raises:
            RateLimitExceededError: If rate limit exceeded
        """
        key = self._get_key(client_id)
        current_time_ms = int(time.time() * 1000)
        window_start_ms = current_time_ms - self.window_ms

        try:
            with redis_client.pipeline() as pipe:
                # Remove expired entries
                pipe.zremrangebyscore(key, 0, window_start_ms)
                # Count current requests in window
                pipe.zcard(key)
                # Add current request
                pipe.zadd(key, {str(current_time_ms): current_time_ms})
                # Set TTL slightly longer than window
                pipe.expire(key, self.window_seconds + 10)

                results = pipe.execute()

            request_count = results[1]  # zcard result

            if request_count >= self.requests_per_minute:
                # Find oldest request to determine retry time
                oldest = redis_client.zrange(key, 0, 0, withscores=True)
                if oldest:
                    oldest_time_ms = int(oldest[0][1])
                    retry_after = max(1, (oldest_time_ms + self.window_ms - current_time_ms) // 1000)
                else:
                    retry_after = self.window_seconds

                raise RateLimitExceededError(
                    limit=self.requests_per_minute,
                    window=retry_after,
                )

            remaining = max(0, self.requests_per_minute - request_count - 1)
            reset_at = int(time.time()) + self.window_seconds

            return RateLimitInfo(
                limit=self.requests_per_minute,
                remaining=remaining,
                reset_at=reset_at,
                concurrent_limit=0,
                concurrent_active=0,
            )

        except RateLimitExceededError:
            raise
        except Exception as e:
            logger.warning(f"Rate limit check failed, allowing request: {e}")
            # Fail open - allow request if Redis is unavailable
            return RateLimitInfo(
                limit=self.requests_per_minute,
                remaining=self.requests_per_minute,
                reset_at=int(time.time()) + self.window_seconds,
                concurrent_limit=0,
                concurrent_active=0,
            )

    def get_status(self, client_id: str) -> RateLimitInfo:
        """Get current rate limit status without recording a request."""
        key = self._get_key(client_id)
        current_time_ms = int(time.time() * 1000)
        window_start_ms = current_time_ms - self.window_ms

        try:
            with redis_client.pipeline() as pipe:
                pipe.zremrangebyscore(key, 0, window_start_ms)
                pipe.zcard(key)
                results = pipe.execute()

            request_count = results[1]
            remaining = max(0, self.requests_per_minute - request_count)

            return RateLimitInfo(
                limit=self.requests_per_minute,
                remaining=remaining,
                reset_at=int(time.time()) + self.window_seconds,
                concurrent_limit=0,
                concurrent_active=0,
            )
        except Exception:
            return RateLimitInfo(
                limit=self.requests_per_minute,
                remaining=self.requests_per_minute,
                reset_at=int(time.time()) + self.window_seconds,
                concurrent_limit=0,
                concurrent_active=0,
            )


class ConcurrentRequestLimiter:
    """
    Limits concurrent active requests using Redis hash maps.

    Features:
    - Track active requests with timestamps
    - Auto-cleanup orphaned requests (crashed workers)
    - Per-client isolation
    """

    KEY_PREFIX = "liteagent:rate_limit"
    REQUEST_MAX_AGE_SECONDS = 600  # 10 minutes - cleanup orphaned requests
    CLEANUP_INTERVAL_SECONDS = 300  # 5 minutes

    def __init__(self, max_concurrent: int | None = None):
        settings = get_settings()
        self.max_concurrent = (
            max_concurrent if max_concurrent is not None
            else settings.rate_limit_max_concurrent
        )
        self._last_cleanup: dict[str, float] = {}

    def _get_key(self, client_id: str) -> str:
        """Generate Redis key for client."""
        return f"{self.KEY_PREFIX}:concurrent:{client_id}"

    def _cleanup_if_needed(self, client_id: str) -> None:
        """Cleanup orphaned requests periodically."""
        key = self._get_key(client_id)
        now = time.time()

        last = self._last_cleanup.get(client_id, 0)
        if now - last < self.CLEANUP_INTERVAL_SECONDS:
            return

        self._last_cleanup[client_id] = now

        try:
            # Get all active requests
            requests = redis_client.hgetall(key)

            # Find requests older than max age
            expired = []
            for request_id, timestamp_bytes in requests.items():
                try:
                    timestamp = float(timestamp_bytes.decode("utf-8"))
                    if now - timestamp > self.REQUEST_MAX_AGE_SECONDS:
                        expired.append(request_id)
                except (ValueError, AttributeError):
                    expired.append(request_id)

            # Remove expired requests
            if expired:
                redis_client.hdel(key, *expired)
                logger.info(
                    f"Cleaned up {len(expired)} orphaned requests for {client_id}"
                )
        except Exception as e:
            logger.warning(f"Concurrent cleanup failed: {e}")

    def enter(self, client_id: str, request_id: str | None = None) -> str:
        """
        Track entering a request.

        Args:
            client_id: Client identifier
            request_id: Optional request ID (generated if not provided)

        Returns:
            Request ID for use with exit()

        Raises:
            ConcurrentLimitExceededError: If at capacity
        """
        if self.max_concurrent <= 0:
            return "unlimited"

        self._cleanup_if_needed(client_id)

        key = self._get_key(client_id)
        request_id = request_id or str(uuid.uuid4())

        try:
            # Check current count
            current_count = redis_client.hlen(key)

            if current_count >= self.max_concurrent:
                raise ConcurrentLimitExceededError(self.max_concurrent)

            # Record request with timestamp
            redis_client.hset(key, request_id, str(time.time()))
            redis_client.expire(key, self.REQUEST_MAX_AGE_SECONDS + 60)

            return request_id

        except ConcurrentLimitExceededError:
            raise
        except Exception as e:
            logger.warning(f"Concurrent enter failed, allowing: {e}")
            return request_id

    def exit(self, client_id: str, request_id: str) -> None:
        """
        Track request completion.

        Args:
            client_id: Client identifier
            request_id: Request ID from enter()
        """
        if request_id == "unlimited":
            return

        key = self._get_key(client_id)

        try:
            redis_client.hdel(key, request_id)
        except Exception as e:
            logger.warning(f"Concurrent exit failed: {e}")

    def get_active_count(self, client_id: str) -> int:
        """Get number of active requests for client."""
        key = self._get_key(client_id)
        try:
            return redis_client.hlen(key)
        except Exception:
            return 0

    @contextmanager
    def request_context(
        self,
        client_id: str,
        request_id: str | None = None,
    ) -> Generator[str, None, None]:
        """
        Context manager for tracking request lifecycle.

        Usage:
            with concurrent_limiter.request_context("client_123") as req_id:
                # Process request
                pass
        """
        req_id = self.enter(client_id, request_id)
        try:
            yield req_id
        finally:
            self.exit(client_id, req_id)


class CombinedRateLimiter:
    """
    Combined rate limiter with both sliding window and concurrent limits.

    Production-ready for multi-tenant chatbot applications.
    """

    def __init__(
        self,
        requests_per_minute: int | None = None,
        max_concurrent: int | None = None,
    ):
        self.sliding_window = SlidingWindowRateLimiter(requests_per_minute)
        self.concurrent = ConcurrentRequestLimiter(max_concurrent)

    def check_rate_limit(self, client_id: str) -> RateLimitInfo:
        """
        Check rate limit (call before processing request).

        Raises:
            RateLimitExceededError: If rate limit exceeded
        """
        return self.sliding_window.check(client_id)

    def enter_request(
        self,
        client_id: str,
        request_id: str | None = None,
    ) -> str:
        """
        Enter a request (call after rate limit check).

        Raises:
            ConcurrentLimitExceededError: If at capacity
        """
        return self.concurrent.enter(client_id, request_id)

    def exit_request(self, client_id: str, request_id: str) -> None:
        """Exit a request (call in finally block)."""
        self.concurrent.exit(client_id, request_id)

    def get_status(self, client_id: str) -> RateLimitInfo:
        """Get complete rate limit status."""
        info = self.sliding_window.get_status(client_id)
        info.concurrent_limit = self.concurrent.max_concurrent
        info.concurrent_active = self.concurrent.get_active_count(client_id)
        return info

    @contextmanager
    def request(
        self,
        client_id: str,
    ) -> Generator[RateLimitInfo, None, None]:
        """
        Full request lifecycle with rate + concurrent limiting.

        Usage:
            with rate_limiter.request("client_123") as info:
                # info contains rate limit status
                process_request()
        """
        # Check rate limit first
        info = self.check_rate_limit(client_id)

        # Then check concurrent limit
        request_id = self.enter_request(client_id)
        info.concurrent_limit = self.concurrent.max_concurrent
        info.concurrent_active = self.concurrent.get_active_count(client_id)

        try:
            yield info
        finally:
            self.exit_request(client_id, request_id)


# Global rate limiter instance
_rate_limiter: CombinedRateLimiter | None = None


def get_rate_limiter() -> CombinedRateLimiter:
    """Get the global rate limiter instance."""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = CombinedRateLimiter()
    return _rate_limiter


def rate_limit_headers(info: RateLimitInfo) -> dict[str, str]:
    """Generate standard rate limit headers."""
    return {
        "X-RateLimit-Limit": str(info.limit),
        "X-RateLimit-Remaining": str(info.remaining),
        "X-RateLimit-Reset": str(info.reset_at),
        "X-RateLimit-Concurrent-Limit": str(info.concurrent_limit),
        "X-RateLimit-Concurrent-Active": str(info.concurrent_active),
    }
