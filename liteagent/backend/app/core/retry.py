"""
Retry with exponential backoff for transient failures.
Provides resilient request handling for external API calls.
"""
import asyncio
import random
from dataclasses import dataclass
from typing import Callable, TypeVar

T = TypeVar("T")


# Default retryable exceptions
RETRYABLE_EXCEPTIONS = (
    ConnectionError,
    TimeoutError,
    OSError,
)


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""

    max_retries: int = 3
    initial_delay_seconds: float = 1.0
    max_delay_seconds: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True


class RetryError(Exception):
    """Raised when all retry attempts are exhausted."""

    def __init__(
        self,
        message: str,
        attempts: int,
        original_error: Exception | None = None,
    ):
        self.attempts = attempts
        self.original_error = original_error
        super().__init__(f"{message} after {attempts} attempts")


class ExponentialBackoff:
    """Calculates exponential backoff delays."""

    def __init__(
        self,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        base: float = 2.0,
        jitter: bool = True,
    ):
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.base = base
        self.jitter = jitter

    def get_delay(self, attempt: int) -> float:
        """
        Calculate delay for a given attempt number.

        Args:
            attempt: Zero-indexed attempt number.

        Returns:
            Delay in seconds.
        """
        delay = self.initial_delay * (self.base ** attempt)
        delay = min(delay, self.max_delay)

        if self.jitter:
            # Add random jitter ±50%
            jitter_factor = 0.5 + random.random()
            delay *= jitter_factor

        return delay


class RetryStrategy:
    """Determines whether to retry based on error and attempt count."""

    def __init__(
        self,
        max_retries: int = 3,
        retryable_exceptions: tuple = RETRYABLE_EXCEPTIONS,
    ):
        self.max_retries = max_retries
        self.retryable_exceptions = retryable_exceptions

    def should_retry(self, error: Exception, attempt: int) -> bool:
        """
        Determine if request should be retried.

        Args:
            error: The exception that occurred.
            attempt: Current attempt number (0-indexed).

        Returns:
            True if should retry.
        """
        if attempt >= self.max_retries:
            return False

        return isinstance(error, self.retryable_exceptions)


def is_retryable_error(error: Exception) -> bool:
    """
    Check if an error is retryable.

    Args:
        error: The exception to check.

    Returns:
        True if the error is retryable.
    """
    return isinstance(error, RETRYABLE_EXCEPTIONS)


async def retry_with_backoff(
    func: Callable[[], T],
    config: RetryConfig | None = None,
    on_retry: Callable[[int, Exception, float], None] | None = None,
) -> T:
    """
    Execute a function with retry and exponential backoff.

    Args:
        func: Async function to execute.
        config: Retry configuration.
        on_retry: Optional callback called before each retry.

    Returns:
        Result of the function.

    Raises:
        RetryError: If all retries are exhausted.
        Exception: Non-retryable exceptions are raised immediately.
    """
    config = config or RetryConfig()

    backoff = ExponentialBackoff(
        initial_delay=config.initial_delay_seconds,
        max_delay=config.max_delay_seconds,
        base=config.exponential_base,
        jitter=config.jitter,
    )

    strategy = RetryStrategy(
        max_retries=config.max_retries,
        retryable_exceptions=RETRYABLE_EXCEPTIONS,
    )

    attempts = 0

    while True:
        try:
            return await func()
        except Exception as e:
            attempts += 1

            if not strategy.should_retry(e, attempts - 1):
                if not is_retryable_error(e):
                    # Non-retryable error, raise immediately
                    raise

                # Retryable but exhausted attempts
                raise RetryError(
                    "All retry attempts exhausted",
                    attempts=attempts,
                    original_error=e,
                )

            delay = backoff.get_delay(attempts - 1)

            if on_retry:
                on_retry(attempts, e, delay)

            await asyncio.sleep(delay)


def retry(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
):
    """
    Decorator to add retry with backoff to an async function.

    Args:
        max_retries: Maximum number of retries.
        initial_delay: Initial delay in seconds.
        max_delay: Maximum delay in seconds.

    Returns:
        Decorated function.
    """
    config = RetryConfig(
        max_retries=max_retries,
        initial_delay_seconds=initial_delay,
        max_delay_seconds=max_delay,
    )

    def decorator(func: Callable):
        async def wrapper(*args, **kwargs):
            async def call_func():
                return await func(*args, **kwargs)

            return await retry_with_backoff(call_func, config=config)

        return wrapper

    return decorator
