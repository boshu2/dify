"""
Sentry integration for error tracking and APM.

Provides error capturing, performance monitoring, and profiling.
"""

import functools
from contextlib import contextmanager
from typing import Any, Callable, TypeVar

from app.core.observability.config import get_observability_config

F = TypeVar("F", bound=Callable[..., Any])


class SentryManager:
    """Manager for Sentry SDK integration."""

    def __init__(
        self,
        dsn: str,
        environment: str = "development",
        release: str | None = None,
        traces_sample_rate: float = 0.1,
        profiles_sample_rate: float = 0.1,
        send_default_pii: bool = False,
        service_name: str = "liteagent",
    ):
        self.dsn = dsn
        self.environment = environment
        self.release = release
        self.traces_sample_rate = traces_sample_rate
        self.profiles_sample_rate = profiles_sample_rate
        self.send_default_pii = send_default_pii
        self.service_name = service_name
        self._initialized = False
        self._sentry_sdk: Any = None

    def initialize(self) -> bool:
        """Initialize Sentry SDK."""
        if self._initialized:
            return True

        if not self.dsn:
            return False

        try:
            import sentry_sdk
            from sentry_sdk.integrations.asyncio import AsyncioIntegration
            from sentry_sdk.integrations.logging import LoggingIntegration

            sentry_sdk.init(
                dsn=self.dsn,
                environment=self.environment,
                release=self.release,
                traces_sample_rate=self.traces_sample_rate,
                profiles_sample_rate=self.profiles_sample_rate,
                send_default_pii=self.send_default_pii,
                integrations=[
                    AsyncioIntegration(),
                    LoggingIntegration(
                        level=None,  # Capture all log levels as breadcrumbs
                        event_level=None,  # Don't send log messages as events
                    ),
                ],
                # Additional options
                attach_stacktrace=True,
                include_local_variables=True,
                max_breadcrumbs=50,
            )

            # Set service name tag
            sentry_sdk.set_tag("service", self.service_name)

            self._sentry_sdk = sentry_sdk
            self._initialized = True
            return True

        except ImportError:
            print("sentry-sdk not installed. Sentry integration disabled.")
            return False
        except Exception as e:
            print(f"Failed to initialize Sentry: {e}")
            return False

    def capture_exception(
        self,
        error: Exception | None = None,
        **kwargs: Any,
    ) -> str | None:
        """Capture an exception to Sentry."""
        if not self._initialized or not self._sentry_sdk:
            return None

        try:
            return self._sentry_sdk.capture_exception(error, **kwargs)
        except Exception:
            return None

    def capture_message(
        self,
        message: str,
        level: str = "info",
        **kwargs: Any,
    ) -> str | None:
        """Capture a message to Sentry."""
        if not self._initialized or not self._sentry_sdk:
            return None

        try:
            return self._sentry_sdk.capture_message(message, level=level, **kwargs)
        except Exception:
            return None

    def set_user(
        self,
        user_id: str | None = None,
        email: str | None = None,
        username: str | None = None,
        **extra: Any,
    ) -> None:
        """Set user context for Sentry."""
        if not self._initialized or not self._sentry_sdk:
            return

        user_data: dict[str, Any] = {}
        if user_id:
            user_data["id"] = user_id
        if email:
            user_data["email"] = email
        if username:
            user_data["username"] = username
        user_data.update(extra)

        if user_data:
            self._sentry_sdk.set_user(user_data)

    def set_tag(self, key: str, value: str) -> None:
        """Set a tag for Sentry context."""
        if not self._initialized or not self._sentry_sdk:
            return
        self._sentry_sdk.set_tag(key, value)

    def set_context(self, name: str, data: dict[str, Any]) -> None:
        """Set additional context for Sentry."""
        if not self._initialized or not self._sentry_sdk:
            return
        self._sentry_sdk.set_context(name, data)

    def add_breadcrumb(
        self,
        category: str,
        message: str,
        level: str = "info",
        data: dict[str, Any] | None = None,
    ) -> None:
        """Add a breadcrumb for debugging."""
        if not self._initialized or not self._sentry_sdk:
            return

        self._sentry_sdk.add_breadcrumb(
            category=category,
            message=message,
            level=level,
            data=data or {},
        )

    @contextmanager
    def start_transaction(
        self,
        name: str,
        op: str = "function",
        **kwargs: Any,
    ):
        """Start a Sentry transaction for performance monitoring."""
        if not self._initialized or not self._sentry_sdk:
            yield None
            return

        transaction = self._sentry_sdk.start_transaction(
            name=name,
            op=op,
            **kwargs,
        )
        try:
            yield transaction
        finally:
            transaction.finish()

    @contextmanager
    def start_span(
        self,
        op: str,
        description: str | None = None,
        **kwargs: Any,
    ):
        """Start a Sentry span within a transaction."""
        if not self._initialized or not self._sentry_sdk:
            yield None
            return

        with self._sentry_sdk.start_span(
            op=op,
            description=description,
            **kwargs,
        ) as span:
            yield span

    def flush(self, timeout: float = 2.0) -> None:
        """Flush pending events to Sentry."""
        if not self._initialized or not self._sentry_sdk:
            return
        self._sentry_sdk.flush(timeout=timeout)

    def close(self) -> None:
        """Close the Sentry client."""
        if self._initialized and self._sentry_sdk:
            self._sentry_sdk.flush(timeout=5.0)


# Global Sentry manager
_sentry_manager: SentryManager | None = None


def get_sentry_manager() -> SentryManager | None:
    """Get the global Sentry manager."""
    global _sentry_manager
    return _sentry_manager


def init_sentry(
    dsn: str | None = None,
    environment: str | None = None,
    **kwargs: Any,
) -> SentryManager | None:
    """Initialize Sentry with configuration."""
    global _sentry_manager

    config = get_observability_config()

    if not config.sentry_enabled:
        return None

    _sentry_manager = SentryManager(
        dsn=dsn or config.sentry_dsn,
        environment=environment or config.environment,
        traces_sample_rate=config.sentry_traces_sample_rate,
        profiles_sample_rate=config.sentry_profiles_sample_rate,
        send_default_pii=config.sentry_send_default_pii,
        service_name=config.service_name,
        **kwargs,
    )

    if _sentry_manager.initialize():
        return _sentry_manager
    return None


def capture_exception(error: Exception | None = None, **kwargs: Any) -> str | None:
    """Capture an exception to Sentry (convenience function)."""
    manager = get_sentry_manager()
    if manager:
        return manager.capture_exception(error, **kwargs)
    return None


def capture_message(message: str, level: str = "info", **kwargs: Any) -> str | None:
    """Capture a message to Sentry (convenience function)."""
    manager = get_sentry_manager()
    if manager:
        return manager.capture_message(message, level, **kwargs)
    return None


def sentry_trace(
    op: str = "function",
    name: str | None = None,
) -> Callable[[F], F]:
    """Decorator for tracing functions with Sentry."""

    def decorator(func: F) -> F:
        span_name = name or func.__name__

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            manager = get_sentry_manager()
            if not manager:
                return func(*args, **kwargs)

            with manager.start_span(op=op, description=span_name):
                return func(*args, **kwargs)

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            manager = get_sentry_manager()
            if not manager:
                return await func(*args, **kwargs)

            with manager.start_span(op=op, description=span_name):
                return await func(*args, **kwargs)

        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return wrapper  # type: ignore

    return decorator
