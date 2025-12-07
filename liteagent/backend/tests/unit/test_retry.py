"""
Unit tests for retry with exponential backoff.
Tests retry logic for transient failures.
"""
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch

from app.core.retry import (
    RetryConfig,
    RetryStrategy,
    ExponentialBackoff,
    retry_with_backoff,
    RetryError,
    is_retryable_error,
)


class TestRetryConfig:
    """Tests for retry configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = RetryConfig()
        assert config.max_retries == 3
        assert config.initial_delay_seconds == 1.0
        assert config.max_delay_seconds == 60.0
        assert config.exponential_base == 2.0

    def test_custom_config(self):
        """Test custom configuration values."""
        config = RetryConfig(
            max_retries=5,
            initial_delay_seconds=0.5,
            max_delay_seconds=30.0,
        )
        assert config.max_retries == 5
        assert config.initial_delay_seconds == 0.5


class TestExponentialBackoff:
    """Tests for exponential backoff calculation."""

    def test_first_retry_delay(self):
        """Test first retry uses initial delay."""
        backoff = ExponentialBackoff(initial_delay=1.0, max_delay=60.0, base=2.0)
        delay = backoff.get_delay(attempt=0)
        assert 0.5 <= delay <= 1.5  # With jitter

    def test_exponential_increase(self):
        """Test delay increases exponentially."""
        backoff = ExponentialBackoff(initial_delay=1.0, max_delay=60.0, base=2.0, jitter=False)
        delay0 = backoff.get_delay(attempt=0)
        delay1 = backoff.get_delay(attempt=1)
        delay2 = backoff.get_delay(attempt=2)

        assert delay0 == 1.0
        assert delay1 == 2.0
        assert delay2 == 4.0

    def test_max_delay_cap(self):
        """Test delay is capped at max_delay."""
        backoff = ExponentialBackoff(initial_delay=1.0, max_delay=10.0, base=2.0, jitter=False)
        delay = backoff.get_delay(attempt=10)  # Would be 1024 without cap
        assert delay == 10.0

    def test_jitter_adds_randomness(self):
        """Test jitter adds randomness to delay."""
        backoff = ExponentialBackoff(initial_delay=1.0, max_delay=60.0, base=2.0, jitter=True)
        delays = [backoff.get_delay(attempt=0) for _ in range(10)]
        # With jitter, delays should vary
        assert len(set(delays)) > 1


class TestRetryStrategy:
    """Tests for retry strategy."""

    def test_should_retry_on_retryable_error(self):
        """Test strategy allows retry for retryable errors."""
        strategy = RetryStrategy()
        error = ConnectionError("Connection refused")
        assert strategy.should_retry(error, attempt=0) is True

    def test_should_not_retry_on_non_retryable_error(self):
        """Test strategy doesn't retry non-retryable errors."""
        strategy = RetryStrategy()
        error = ValueError("Invalid input")
        assert strategy.should_retry(error, attempt=0) is False

    def test_should_not_retry_after_max_attempts(self):
        """Test strategy stops after max retries."""
        strategy = RetryStrategy(max_retries=3)
        error = ConnectionError("Connection refused")
        assert strategy.should_retry(error, attempt=3) is False

    def test_custom_retryable_exceptions(self):
        """Test custom retryable exceptions."""
        strategy = RetryStrategy(
            retryable_exceptions=(ValueError,),
            max_retries=3,
        )
        assert strategy.should_retry(ValueError("test"), attempt=0) is True
        assert strategy.should_retry(TypeError("test"), attempt=0) is False


class TestIsRetryableError:
    """Tests for retryable error detection."""

    def test_connection_error_is_retryable(self):
        """Test ConnectionError is retryable."""
        assert is_retryable_error(ConnectionError()) is True

    def test_timeout_error_is_retryable(self):
        """Test TimeoutError is retryable."""
        assert is_retryable_error(TimeoutError()) is True

    def test_value_error_is_not_retryable(self):
        """Test ValueError is not retryable."""
        assert is_retryable_error(ValueError()) is False


class TestRetryWithBackoff:
    """Tests for retry with backoff function."""

    @pytest.mark.asyncio
    async def test_succeeds_on_first_try(self):
        """Test function that succeeds immediately."""
        call_count = 0

        async def success_func():
            nonlocal call_count
            call_count += 1
            return "success"

        result = await retry_with_backoff(success_func)
        assert result == "success"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retries_on_failure(self):
        """Test retries on transient failure."""
        call_count = 0

        async def fail_then_succeed():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Transient error")
            return "success"

        config = RetryConfig(max_retries=5, initial_delay_seconds=0.01)
        result = await retry_with_backoff(fail_then_succeed, config=config)

        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_raises_after_max_retries(self):
        """Test raises error after max retries exhausted."""
        call_count = 0

        async def always_fail():
            nonlocal call_count
            call_count += 1
            raise ConnectionError("Persistent error")

        config = RetryConfig(max_retries=3, initial_delay_seconds=0.01)

        with pytest.raises(RetryError) as exc_info:
            await retry_with_backoff(always_fail, config=config)

        assert call_count == 4  # Initial + 3 retries
        assert exc_info.value.attempts == 4

    @pytest.mark.asyncio
    async def test_no_retry_for_non_retryable_error(self):
        """Test non-retryable errors are raised immediately."""
        call_count = 0

        async def raise_value_error():
            nonlocal call_count
            call_count += 1
            raise ValueError("Invalid input")

        with pytest.raises(ValueError):
            await retry_with_backoff(raise_value_error)

        assert call_count == 1

    @pytest.mark.asyncio
    async def test_on_retry_callback(self):
        """Test on_retry callback is called."""
        retry_attempts = []

        async def fail_twice():
            if len(retry_attempts) < 2:
                raise ConnectionError("Transient")
            return "success"

        def on_retry(attempt, error, delay):
            retry_attempts.append((attempt, str(error)))

        config = RetryConfig(max_retries=3, initial_delay_seconds=0.01)
        await retry_with_backoff(fail_twice, config=config, on_retry=on_retry)

        assert len(retry_attempts) == 2


class TestRetryError:
    """Tests for retry error."""

    def test_error_contains_attempts(self):
        """Test error contains attempt count."""
        error = RetryError("All retries failed", attempts=4)
        assert error.attempts == 4
        assert "4" in str(error)

    def test_error_contains_original_error(self):
        """Test error contains original exception."""
        original = ConnectionError("Connection failed")
        error = RetryError("Retry failed", attempts=3, original_error=original)
        assert error.original_error is original
