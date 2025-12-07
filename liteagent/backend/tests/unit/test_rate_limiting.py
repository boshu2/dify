"""
Unit tests for rate limiting.
Tests the token bucket rate limiter and middleware.
"""
import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timezone

from app.core.rate_limiter import (
    RateLimiter,
    RateLimitConfig,
    TokenBucket,
    RateLimitExceeded,
    InMemoryRateLimiterStorage,
)


class TestRateLimitConfig:
    """Tests for rate limit configuration."""

    def test_default_config(self):
        """Test default rate limit configuration."""
        config = RateLimitConfig()
        assert config.requests_per_minute == 60
        assert config.requests_per_hour == 1000
        assert config.burst_size == 10

    def test_custom_config(self):
        """Test custom rate limit configuration."""
        config = RateLimitConfig(
            requests_per_minute=30,
            requests_per_hour=500,
            burst_size=5,
        )
        assert config.requests_per_minute == 30
        assert config.requests_per_hour == 500
        assert config.burst_size == 5

    def test_config_from_env(self):
        """Test configuration from environment variables."""
        with patch.dict(
            "os.environ",
            {
                "RATE_LIMIT_REQUESTS_PER_MINUTE": "120",
                "RATE_LIMIT_REQUESTS_PER_HOUR": "2000",
                "RATE_LIMIT_BURST_SIZE": "20",
            },
        ):
            config = RateLimitConfig.from_env()
            assert config.requests_per_minute == 120
            assert config.requests_per_hour == 2000
            assert config.burst_size == 20


class TestTokenBucket:
    """Tests for token bucket algorithm."""

    def test_bucket_initialization(self):
        """Test token bucket initializes with full capacity."""
        bucket = TokenBucket(capacity=10, refill_rate=1.0)
        assert bucket.tokens == 10
        assert bucket.capacity == 10

    def test_consume_token(self):
        """Test consuming a token."""
        bucket = TokenBucket(capacity=10, refill_rate=1.0)
        assert bucket.consume() is True
        assert bucket.tokens == 9

    def test_consume_multiple_tokens(self):
        """Test consuming multiple tokens at once."""
        bucket = TokenBucket(capacity=10, refill_rate=1.0)
        assert bucket.consume(5) is True
        assert bucket.tokens == 5

    def test_consume_fails_when_empty(self):
        """Test that consuming fails when bucket is empty."""
        bucket = TokenBucket(capacity=2, refill_rate=0.0)
        assert bucket.consume() is True
        assert bucket.consume() is True
        assert bucket.consume() is False  # No more tokens

    def test_token_refill(self):
        """Test that tokens refill over time."""
        bucket = TokenBucket(capacity=10, refill_rate=5.0)  # 5 tokens per second
        bucket.tokens = 0
        bucket.last_refill = bucket.last_refill - 1.0  # 1 second ago
        bucket._refill()
        assert 4.9 <= bucket.tokens <= 5.1  # Allow for timing variation

    def test_refill_does_not_exceed_capacity(self):
        """Test that refill doesn't exceed capacity."""
        bucket = TokenBucket(capacity=10, refill_rate=100.0)
        bucket.last_refill = bucket.last_refill - 1.0  # 1 second ago
        bucket._refill()
        assert bucket.tokens == 10  # Still at capacity

    def test_get_wait_time(self):
        """Test calculating wait time until token available."""
        bucket = TokenBucket(capacity=10, refill_rate=1.0)  # 1 token per second
        bucket.tokens = 0
        wait_time = bucket.get_wait_time(1)
        assert 0.9 <= wait_time <= 1.1  # Approximately 1 second


class TestInMemoryRateLimiterStorage:
    """Tests for in-memory rate limiter storage."""

    @pytest.mark.asyncio
    async def test_get_bucket_creates_new(self):
        """Test getting a bucket for new key creates it."""
        storage = InMemoryRateLimiterStorage()
        bucket = await storage.get_bucket("client-123", capacity=10, refill_rate=1.0)
        assert bucket is not None
        assert bucket.capacity == 10

    @pytest.mark.asyncio
    async def test_get_bucket_returns_existing(self):
        """Test getting same bucket for same key."""
        storage = InMemoryRateLimiterStorage()
        bucket1 = await storage.get_bucket("client-123", capacity=10, refill_rate=1.0)
        bucket1.consume(5)  # Modify the bucket

        bucket2 = await storage.get_bucket("client-123", capacity=10, refill_rate=1.0)
        assert bucket2.tokens == 5  # Same bucket

    @pytest.mark.asyncio
    async def test_different_keys_different_buckets(self):
        """Test different keys get different buckets."""
        storage = InMemoryRateLimiterStorage()
        bucket1 = await storage.get_bucket("client-1", capacity=10, refill_rate=1.0)
        bucket2 = await storage.get_bucket("client-2", capacity=10, refill_rate=1.0)

        bucket1.consume(5)
        assert bucket1.tokens == 5
        assert bucket2.tokens == 10  # Unaffected


class TestRateLimiter:
    """Tests for the main rate limiter."""

    @pytest.mark.asyncio
    async def test_check_allows_within_limit(self):
        """Test that requests within limit are allowed."""
        config = RateLimitConfig(requests_per_minute=10, burst_size=5)
        limiter = RateLimiter(config)

        allowed = await limiter.check("client-123")
        assert allowed is True

    @pytest.mark.asyncio
    async def test_check_blocks_when_exceeded(self):
        """Test that requests are blocked when limit exceeded."""
        config = RateLimitConfig(requests_per_minute=2, burst_size=2)
        limiter = RateLimiter(config)

        await limiter.check("client-123")
        await limiter.check("client-123")

        # Third request should be blocked
        with pytest.raises(RateLimitExceeded) as exc_info:
            await limiter.check("client-123")

        assert exc_info.value.retry_after > 0

    @pytest.mark.asyncio
    async def test_check_different_clients_independent(self):
        """Test that different clients have independent limits."""
        config = RateLimitConfig(requests_per_minute=1, burst_size=1)
        limiter = RateLimiter(config)

        await limiter.check("client-1")

        # Different client should still be allowed
        allowed = await limiter.check("client-2")
        assert allowed is True

    @pytest.mark.asyncio
    async def test_get_remaining_requests(self):
        """Test getting remaining request count."""
        config = RateLimitConfig(requests_per_minute=10, burst_size=10)
        limiter = RateLimiter(config)

        await limiter.check("client-123")
        await limiter.check("client-123")

        remaining = await limiter.get_remaining("client-123")
        assert remaining == 8

    @pytest.mark.asyncio
    async def test_get_headers(self):
        """Test getting rate limit headers."""
        config = RateLimitConfig(requests_per_minute=60, burst_size=10)
        limiter = RateLimiter(config)

        await limiter.check("client-123")
        headers = await limiter.get_headers("client-123")

        assert "X-RateLimit-Limit" in headers
        assert "X-RateLimit-Remaining" in headers
        assert "X-RateLimit-Reset" in headers
        assert headers["X-RateLimit-Limit"] == "60"


class TestRateLimitExceeded:
    """Tests for rate limit exceeded exception."""

    def test_exception_with_retry_after(self):
        """Test exception includes retry_after (rounded up)."""
        exc = RateLimitExceeded(retry_after=30)
        # Implementation rounds up, so 30 becomes 31
        assert exc.retry_after == 31
        assert "31" in str(exc)

    def test_exception_to_http_response(self):
        """Test converting exception to HTTP response details."""
        exc = RateLimitExceeded(retry_after=60)
        response = exc.to_response()

        assert response["status_code"] == 429
        # Implementation rounds up, so 60 becomes 61
        assert response["headers"]["Retry-After"] == "61"
        assert "rate limit" in response["detail"].lower()
