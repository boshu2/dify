"""
Circuit breaker pattern for resilient external API calls.
Prevents cascading failures by failing fast when services are unavailable.
"""
import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, TypeVar

T = TypeVar("T")


class CircuitState(str, Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation, requests pass through
    OPEN = "open"  # Failing, requests are rejected
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""

    failure_threshold: int = 5  # Failures before opening
    success_threshold: int = 2  # Successes to close from half-open
    timeout_seconds: float = 30  # Time before trying half-open
    half_open_max_calls: int = 3  # Max calls in half-open state
    excluded_exceptions: tuple = ()  # Exceptions that don't count as failures


class CircuitOpenError(Exception):
    """Raised when circuit is open and rejecting requests."""

    def __init__(self, service: str, retry_after: float | None = None):
        self.service = service
        self.retry_after = retry_after
        super().__init__(f"Circuit breaker open for service: {service}")


class CircuitBreaker:
    """
    Circuit breaker implementation.

    States:
    - CLOSED: Normal operation, failures are counted
    - OPEN: Circuit tripped, all calls fail fast
    - HALF_OPEN: Testing if service recovered, limited calls allowed
    """

    def __init__(self, name: str, config: CircuitBreakerConfig | None = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: float | None = None
        self._half_open_calls = 0
        self._lock = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        """Get current circuit state, handling timeout transitions."""
        if self._state == CircuitState.OPEN:
            # Check if timeout has elapsed
            if self._last_failure_time is not None:
                elapsed = time.monotonic() - self._last_failure_time
                if elapsed >= self.config.timeout_seconds:
                    self._state = CircuitState.HALF_OPEN
                    self._half_open_calls = 0
                    self._success_count = 0
        return self._state

    @property
    def failure_count(self) -> int:
        """Get current failure count."""
        return self._failure_count

    async def call(self, func: Callable[[], T]) -> T:
        """
        Execute a function with circuit breaker protection.

        Args:
            func: Async function to execute.

        Returns:
            Result of the function.

        Raises:
            CircuitOpenError: If circuit is open.
            Exception: Any exception from the wrapped function.
        """
        async with self._lock:
            current_state = self.state

            if current_state == CircuitState.OPEN:
                retry_after = None
                if self._last_failure_time:
                    elapsed = time.monotonic() - self._last_failure_time
                    retry_after = max(0, self.config.timeout_seconds - elapsed)
                raise CircuitOpenError(self.name, retry_after=retry_after)

            if current_state == CircuitState.HALF_OPEN:
                if self._half_open_calls >= self.config.half_open_max_calls:
                    raise CircuitOpenError(self.name)
                self._half_open_calls += 1

        try:
            result = await func()
            await self._on_success()
            return result
        except Exception as e:
            # Check if exception should be excluded
            if isinstance(e, self.config.excluded_exceptions):
                raise
            await self._on_failure()
            raise

    async def _on_success(self) -> None:
        """Handle successful call."""
        async with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.config.success_threshold:
                    self._close_circuit()
            else:
                # Reset failure count on success in closed state
                self._failure_count = 0

    async def _on_failure(self) -> None:
        """Handle failed call."""
        async with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.monotonic()

            if self._state == CircuitState.HALF_OPEN:
                self._open_circuit()
            elif self._failure_count >= self.config.failure_threshold:
                self._open_circuit()

    def _open_circuit(self) -> None:
        """Transition to open state."""
        self._state = CircuitState.OPEN
        self._half_open_calls = 0

    def _close_circuit(self) -> None:
        """Transition to closed state."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._half_open_calls = 0

    def get_stats(self) -> dict[str, Any]:
        """Get current circuit breaker statistics."""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
            "last_failure_time": self._last_failure_time,
            "half_open_calls": self._half_open_calls,
        }

    def reset(self) -> None:
        """Reset circuit breaker to initial state."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = None
        self._half_open_calls = 0


class CircuitBreakerRegistry:
    """Registry for managing multiple circuit breakers."""

    def __init__(self, default_config: CircuitBreakerConfig | None = None):
        self._breakers: dict[str, CircuitBreaker] = {}
        self._default_config = default_config or CircuitBreakerConfig()
        self._lock = asyncio.Lock()

    def get_or_create(
        self,
        name: str,
        config: CircuitBreakerConfig | None = None,
    ) -> CircuitBreaker:
        """
        Get or create a circuit breaker for a service.

        Args:
            name: Service name.
            config: Optional custom configuration.

        Returns:
            Circuit breaker instance.
        """
        if name not in self._breakers:
            self._breakers[name] = CircuitBreaker(
                name,
                config or self._default_config,
            )
        return self._breakers[name]

    def get(self, name: str) -> CircuitBreaker | None:
        """Get an existing circuit breaker."""
        return self._breakers.get(name)

    def get_all_stats(self) -> dict[str, dict[str, Any]]:
        """Get statistics for all circuit breakers."""
        return {name: breaker.get_stats() for name, breaker in self._breakers.items()}

    def reset_all(self) -> None:
        """Reset all circuit breakers."""
        for breaker in self._breakers.values():
            breaker.reset()


# Global registry instance
_registry: CircuitBreakerRegistry | None = None


def get_circuit_breaker_registry() -> CircuitBreakerRegistry:
    """Get the global circuit breaker registry."""
    global _registry
    if _registry is None:
        _registry = CircuitBreakerRegistry()
    return _registry


def circuit_breaker(name: str, config: CircuitBreakerConfig | None = None):
    """
    Decorator to wrap a function with circuit breaker protection.

    Args:
        name: Service name for the circuit breaker.
        config: Optional configuration.

    Returns:
        Decorated function.
    """
    def decorator(func: Callable):
        async def wrapper(*args, **kwargs):
            registry = get_circuit_breaker_registry()
            breaker = registry.get_or_create(name, config)

            async def call_func():
                return await func(*args, **kwargs)

            return await breaker.call(call_func)

        return wrapper

    return decorator
