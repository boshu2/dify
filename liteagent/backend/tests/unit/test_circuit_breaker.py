"""
Unit tests for circuit breaker pattern.
Tests state transitions, failure thresholds, and recovery.
"""
import pytest
import asyncio
from datetime import timedelta
from unittest.mock import Mock, AsyncMock, patch

from app.core.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
    CircuitOpenError,
    CircuitBreakerRegistry,
)


class TestCircuitBreakerConfig:
    """Tests for circuit breaker configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = CircuitBreakerConfig()
        assert config.failure_threshold == 5
        assert config.success_threshold == 2
        assert config.timeout_seconds == 30
        assert config.half_open_max_calls == 3

    def test_custom_config(self):
        """Test custom configuration values."""
        config = CircuitBreakerConfig(
            failure_threshold=10,
            success_threshold=3,
            timeout_seconds=60,
        )
        assert config.failure_threshold == 10
        assert config.timeout_seconds == 60


class TestCircuitState:
    """Tests for circuit state enum."""

    def test_state_values(self):
        """Test circuit state values."""
        assert CircuitState.CLOSED.value == "closed"
        assert CircuitState.OPEN.value == "open"
        assert CircuitState.HALF_OPEN.value == "half_open"


class TestCircuitBreaker:
    """Tests for circuit breaker functionality."""

    @pytest.fixture
    def breaker(self):
        """Create a circuit breaker with low thresholds for testing."""
        config = CircuitBreakerConfig(
            failure_threshold=3,
            success_threshold=2,
            timeout_seconds=1,
        )
        return CircuitBreaker("test-service", config)

    def test_initial_state_is_closed(self, breaker):
        """Test circuit starts in closed state."""
        assert breaker.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_successful_call(self, breaker):
        """Test successful call passes through."""
        async def success_func():
            return "success"

        result = await breaker.call(success_func)
        assert result == "success"
        assert breaker.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_failed_call_increments_failure_count(self, breaker):
        """Test failures are counted."""
        async def fail_func():
            raise RuntimeError("Test error")

        with pytest.raises(RuntimeError):
            await breaker.call(fail_func)

        assert breaker.failure_count == 1
        assert breaker.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_circuit_opens_after_threshold(self, breaker):
        """Test circuit opens after failure threshold."""
        async def fail_func():
            raise RuntimeError("Test error")

        # Trigger failures up to threshold
        for _ in range(3):
            with pytest.raises(RuntimeError):
                await breaker.call(fail_func)

        assert breaker.state == CircuitState.OPEN
        assert breaker.failure_count == 3

    @pytest.mark.asyncio
    async def test_open_circuit_rejects_calls(self, breaker):
        """Test open circuit rejects calls immediately."""
        async def fail_func():
            raise RuntimeError("Test error")

        # Open the circuit
        for _ in range(3):
            with pytest.raises(RuntimeError):
                await breaker.call(fail_func)

        # Next call should be rejected
        async def any_func():
            return "should not run"

        with pytest.raises(CircuitOpenError):
            await breaker.call(any_func)

    @pytest.mark.asyncio
    async def test_circuit_transitions_to_half_open(self, breaker):
        """Test circuit transitions to half-open after timeout."""
        async def fail_func():
            raise RuntimeError("Test error")

        # Open the circuit
        for _ in range(3):
            with pytest.raises(RuntimeError):
                await breaker.call(fail_func)

        assert breaker.state == CircuitState.OPEN

        # Wait for timeout
        await asyncio.sleep(1.1)

        # Circuit should be half-open now
        assert breaker.state == CircuitState.HALF_OPEN

    @pytest.mark.asyncio
    async def test_half_open_success_closes_circuit(self, breaker):
        """Test successful calls in half-open state close circuit."""
        async def fail_func():
            raise RuntimeError("Test error")

        async def success_func():
            return "success"

        # Open the circuit
        for _ in range(3):
            with pytest.raises(RuntimeError):
                await breaker.call(fail_func)

        # Wait for timeout
        await asyncio.sleep(1.1)

        # Successful calls in half-open should close circuit
        for _ in range(2):  # success_threshold = 2
            await breaker.call(success_func)

        assert breaker.state == CircuitState.CLOSED
        assert breaker.failure_count == 0

    @pytest.mark.asyncio
    async def test_half_open_failure_reopens_circuit(self, breaker):
        """Test failure in half-open state reopens circuit."""
        async def fail_func():
            raise RuntimeError("Test error")

        # Open the circuit
        for _ in range(3):
            with pytest.raises(RuntimeError):
                await breaker.call(fail_func)

        # Wait for timeout
        await asyncio.sleep(1.1)
        assert breaker.state == CircuitState.HALF_OPEN

        # Failure in half-open should reopen
        with pytest.raises(RuntimeError):
            await breaker.call(fail_func)

        assert breaker.state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_success_resets_failure_count(self, breaker):
        """Test successful call resets failure count."""
        async def fail_func():
            raise RuntimeError("Test error")

        async def success_func():
            return "success"

        # Some failures
        for _ in range(2):
            with pytest.raises(RuntimeError):
                await breaker.call(fail_func)

        assert breaker.failure_count == 2

        # Success resets count
        await breaker.call(success_func)
        assert breaker.failure_count == 0

    def test_get_stats(self, breaker):
        """Test getting circuit breaker statistics."""
        stats = breaker.get_stats()

        assert "state" in stats
        assert "failure_count" in stats
        assert "success_count" in stats
        assert "last_failure_time" in stats
        assert stats["state"] == "closed"


class TestCircuitBreakerRegistry:
    """Tests for circuit breaker registry."""

    def test_get_or_create_breaker(self):
        """Test getting or creating a circuit breaker."""
        registry = CircuitBreakerRegistry()

        breaker1 = registry.get_or_create("service-a")
        breaker2 = registry.get_or_create("service-a")

        assert breaker1 is breaker2  # Same instance

    def test_different_services_different_breakers(self):
        """Test different services get different breakers."""
        registry = CircuitBreakerRegistry()

        breaker_a = registry.get_or_create("service-a")
        breaker_b = registry.get_or_create("service-b")

        assert breaker_a is not breaker_b

    def test_custom_config_per_service(self):
        """Test custom config per service."""
        registry = CircuitBreakerRegistry()
        custom_config = CircuitBreakerConfig(failure_threshold=10)

        breaker = registry.get_or_create("custom-service", config=custom_config)

        assert breaker.config.failure_threshold == 10

    def test_get_all_stats(self):
        """Test getting stats for all breakers."""
        registry = CircuitBreakerRegistry()
        registry.get_or_create("service-a")
        registry.get_or_create("service-b")

        all_stats = registry.get_all_stats()

        assert "service-a" in all_stats
        assert "service-b" in all_stats


class TestCircuitOpenError:
    """Tests for circuit open error."""

    def test_error_message(self):
        """Test error contains service name."""
        error = CircuitOpenError("test-service")
        assert "test-service" in str(error)

    def test_error_attributes(self):
        """Test error has correct attributes."""
        error = CircuitOpenError("test-service", retry_after=30)
        assert error.service == "test-service"
        assert error.retry_after == 30
