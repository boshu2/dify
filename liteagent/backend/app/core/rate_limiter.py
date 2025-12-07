"""
Rate limiting for LiteAgent API.
Uses token bucket algorithm for efficient rate limiting.
"""
import os
import time
import asyncio
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""

    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    burst_size: int = 10

    @classmethod
    def from_env(cls) -> "RateLimitConfig":
        """Create configuration from environment variables."""
        return cls(
            requests_per_minute=int(os.environ.get("RATE_LIMIT_REQUESTS_PER_MINUTE", "60")),
            requests_per_hour=int(os.environ.get("RATE_LIMIT_REQUESTS_PER_HOUR", "1000")),
            burst_size=int(os.environ.get("RATE_LIMIT_BURST_SIZE", "10")),
        )


class TokenBucket:
    """
    Token bucket rate limiter.

    Allows bursts of traffic while maintaining a long-term average rate.
    """

    def __init__(self, capacity: int, refill_rate: float):
        """
        Initialize token bucket.

        Args:
            capacity: Maximum number of tokens (burst size)
            refill_rate: Tokens added per second
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = float(capacity)
        self.last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self.last_refill
        tokens_to_add = elapsed * self.refill_rate

        self.tokens = min(self.capacity, self.tokens + tokens_to_add)
        self.last_refill = now

    def consume(self, tokens: int = 1) -> bool:
        """
        Try to consume tokens from the bucket.

        Args:
            tokens: Number of tokens to consume.

        Returns:
            True if tokens were consumed, False if not enough tokens.
        """
        self._refill()

        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False

    def get_wait_time(self, tokens: int = 1) -> float:
        """
        Get time to wait until tokens are available.

        Args:
            tokens: Number of tokens needed.

        Returns:
            Seconds to wait (0 if tokens are available now).
        """
        self._refill()

        if self.tokens >= tokens:
            return 0.0

        tokens_needed = tokens - self.tokens
        if self.refill_rate <= 0:
            return float("inf")
        return tokens_needed / self.refill_rate


class RateLimitExceeded(Exception):
    """Exception raised when rate limit is exceeded."""

    def __init__(self, retry_after: float, message: str | None = None):
        self.retry_after = int(retry_after) + 1  # Round up
        self.message = message or f"Rate limit exceeded. Retry after {self.retry_after} seconds."
        super().__init__(self.message)

    def to_response(self) -> dict[str, Any]:
        """Convert to HTTP response details."""
        return {
            "status_code": 429,
            "detail": self.message,
            "headers": {
                "Retry-After": str(self.retry_after),
            },
        }


class InMemoryRateLimiterStorage:
    """In-memory storage for rate limiter buckets."""

    def __init__(self):
        self._buckets: dict[str, TokenBucket] = {}
        self._lock = asyncio.Lock()

    async def get_bucket(
        self,
        key: str,
        capacity: int,
        refill_rate: float,
    ) -> TokenBucket:
        """
        Get or create a token bucket for the given key.

        Args:
            key: Client identifier (IP, API key, etc.)
            capacity: Bucket capacity
            refill_rate: Token refill rate

        Returns:
            Token bucket for the client.
        """
        async with self._lock:
            if key not in self._buckets:
                self._buckets[key] = TokenBucket(capacity, refill_rate)
            return self._buckets[key]

    async def cleanup_expired(self, max_age_seconds: float = 3600) -> int:
        """
        Remove buckets that haven't been used recently.

        Args:
            max_age_seconds: Remove buckets older than this.

        Returns:
            Number of buckets removed.
        """
        now = time.monotonic()
        to_remove = []

        async with self._lock:
            for key, bucket in self._buckets.items():
                if now - bucket.last_refill > max_age_seconds:
                    to_remove.append(key)

            for key in to_remove:
                del self._buckets[key]

        return len(to_remove)


class RateLimiter:
    """
    Rate limiter using token bucket algorithm.

    Provides per-client rate limiting with configurable limits.
    """

    def __init__(
        self,
        config: RateLimitConfig | None = None,
        storage: InMemoryRateLimiterStorage | None = None,
    ):
        """
        Initialize rate limiter.

        Args:
            config: Rate limit configuration.
            storage: Storage backend for buckets.
        """
        self.config = config or RateLimitConfig()
        self.storage = storage or InMemoryRateLimiterStorage()

        # Calculate refill rate: tokens per second based on requests per minute
        self.refill_rate = self.config.requests_per_minute / 60.0

    async def check(self, client_id: str) -> bool:
        """
        Check if client can make a request.

        Args:
            client_id: Client identifier.

        Returns:
            True if request is allowed.

        Raises:
            RateLimitExceeded: If rate limit is exceeded.
        """
        bucket = await self.storage.get_bucket(
            client_id,
            capacity=self.config.burst_size,
            refill_rate=self.refill_rate,
        )

        if bucket.consume():
            return True

        wait_time = bucket.get_wait_time()
        raise RateLimitExceeded(retry_after=wait_time)

    async def get_remaining(self, client_id: str) -> int:
        """
        Get remaining requests for a client.

        Args:
            client_id: Client identifier.

        Returns:
            Number of remaining requests.
        """
        bucket = await self.storage.get_bucket(
            client_id,
            capacity=self.config.burst_size,
            refill_rate=self.refill_rate,
        )
        bucket._refill()
        return int(bucket.tokens)

    async def get_headers(self, client_id: str) -> dict[str, str]:
        """
        Get rate limit headers for response.

        Args:
            client_id: Client identifier.

        Returns:
            Dict of HTTP headers to include in response.
        """
        remaining = await self.get_remaining(client_id)
        bucket = await self.storage.get_bucket(
            client_id,
            capacity=self.config.burst_size,
            refill_rate=self.refill_rate,
        )

        # Calculate reset time (when bucket will be full again)
        tokens_needed = self.config.burst_size - bucket.tokens
        if tokens_needed > 0 and self.refill_rate > 0:
            reset_in = tokens_needed / self.refill_rate
        else:
            reset_in = 0

        return {
            "X-RateLimit-Limit": str(self.config.requests_per_minute),
            "X-RateLimit-Remaining": str(remaining),
            "X-RateLimit-Reset": str(int(time.time() + reset_in)),
        }


# Global rate limiter instance
_rate_limiter: RateLimiter | None = None


def get_rate_limiter() -> RateLimiter:
    """Get the global rate limiter instance."""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter(RateLimitConfig.from_env())
    return _rate_limiter
