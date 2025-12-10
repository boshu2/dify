"""
Tracer implementation for LiteAgent.

Provides a unified interface for tracing that can export to multiple backends.
"""

import asyncio
import functools
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, TypeVar

from app.core.observability.context import (
    TraceContext,
    get_current_context,
    set_trace_context,
    trace_context,
)

F = TypeVar("F", bound=Callable[..., Any])


class SpanKind(str, Enum):
    """Types of spans."""

    INTERNAL = "internal"
    SERVER = "server"
    CLIENT = "client"
    PRODUCER = "producer"
    CONSUMER = "consumer"


class SpanStatus(str, Enum):
    """Span completion status."""

    UNSET = "unset"
    OK = "ok"
    ERROR = "error"


@dataclass
class SpanEvent:
    """Event recorded during a span."""

    name: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class Span:
    """Represents a single trace span."""

    name: str
    trace_id: str
    span_id: str
    parent_span_id: str | None = None
    kind: SpanKind = SpanKind.INTERNAL
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    end_time: datetime | None = None
    status: SpanStatus = SpanStatus.UNSET
    status_message: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[SpanEvent] = field(default_factory=list)

    # LLM-specific attributes
    model: str | None = None
    provider: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None

    # Tool-specific attributes
    tool_name: str | None = None
    tool_input: dict[str, Any] | None = None
    tool_output: Any | None = None

    def set_attribute(self, key: str, value: Any) -> None:
        """Set a span attribute."""
        self.attributes[key] = value

    def set_attributes(self, attributes: dict[str, Any]) -> None:
        """Set multiple span attributes."""
        self.attributes.update(attributes)

    def add_event(
        self,
        name: str,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        """Add an event to the span."""
        self.events.append(SpanEvent(
            name=name,
            attributes=attributes or {},
        ))

    def set_status(self, status: SpanStatus, message: str | None = None) -> None:
        """Set the span status."""
        self.status = status
        self.status_message = message

    def set_ok(self) -> None:
        """Mark span as successful."""
        self.status = SpanStatus.OK

    def set_error(self, message: str | None = None) -> None:
        """Mark span as failed."""
        self.status = SpanStatus.ERROR
        self.status_message = message

    def end(self) -> None:
        """End the span."""
        self.end_time = datetime.now(timezone.utc)

    @property
    def duration_ms(self) -> float | None:
        """Get span duration in milliseconds."""
        if self.end_time:
            delta = self.end_time - self.start_time
            return delta.total_seconds() * 1000
        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for export."""
        data: dict[str, Any] = {
            "name": self.name,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "kind": self.kind.value,
            "start_time": self.start_time.isoformat(),
            "status": self.status.value,
            "attributes": self.attributes,
        }

        if self.end_time:
            data["end_time"] = self.end_time.isoformat()
            data["duration_ms"] = self.duration_ms

        if self.status_message:
            data["status_message"] = self.status_message

        if self.events:
            data["events"] = [
                {
                    "name": e.name,
                    "timestamp": e.timestamp.isoformat(),
                    "attributes": e.attributes,
                }
                for e in self.events
            ]

        # Add LLM attributes if present
        if self.model:
            data["model"] = self.model
        if self.provider:
            data["provider"] = self.provider
        if self.prompt_tokens is not None:
            data["prompt_tokens"] = self.prompt_tokens
        if self.completion_tokens is not None:
            data["completion_tokens"] = self.completion_tokens
        if self.total_tokens is not None:
            data["total_tokens"] = self.total_tokens

        # Add tool attributes if present
        if self.tool_name:
            data["tool_name"] = self.tool_name
        if self.tool_input is not None:
            data["tool_input"] = self.tool_input
        if self.tool_output is not None:
            data["tool_output"] = self.tool_output

        return data


class SpanExporter(ABC):
    """Base class for span exporters."""

    @abstractmethod
    async def export(self, spans: list[Span]) -> bool:
        """Export spans to the backend."""
        pass

    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the exporter."""
        pass


class InMemoryExporter(SpanExporter):
    """In-memory exporter for testing."""

    def __init__(self) -> None:
        self.spans: list[Span] = []

    async def export(self, spans: list[Span]) -> bool:
        self.spans.extend(spans)
        return True

    async def shutdown(self) -> None:
        pass

    def clear(self) -> None:
        self.spans.clear()


class ConsoleExporter(SpanExporter):
    """Console exporter for debugging."""

    async def export(self, spans: list[Span]) -> bool:
        import json

        for span in spans:
            print(json.dumps(span.to_dict(), indent=2, default=str))
        return True

    async def shutdown(self) -> None:
        pass


class Tracer:
    """Main tracer class for creating and managing spans."""

    def __init__(
        self,
        service_name: str = "liteagent",
        exporters: list[SpanExporter] | None = None,
        enabled: bool = True,
    ):
        self.service_name = service_name
        self.exporters = exporters or []
        self.enabled = enabled
        self._pending_spans: list[Span] = []
        self._batch_size = 100
        self._flush_interval = 5.0  # seconds

    def add_exporter(self, exporter: SpanExporter) -> None:
        """Add an exporter."""
        self.exporters.append(exporter)

    @contextmanager
    def start_span(
        self,
        name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: dict[str, Any] | None = None,
    ):
        """Start a new span as a context manager."""
        if not self.enabled:
            yield None
            return

        # Get or create trace context
        parent_ctx = get_current_context()
        if parent_ctx:
            ctx = parent_ctx.create_child()
        else:
            ctx = TraceContext()

        span = Span(
            name=name,
            trace_id=ctx.trace_id,
            span_id=ctx.span_id,
            parent_span_id=ctx.parent_span_id,
            kind=kind,
            attributes=attributes or {},
        )

        # Add service name
        span.set_attribute("service.name", self.service_name)

        with trace_context(ctx):
            try:
                yield span
                if span.status == SpanStatus.UNSET:
                    span.set_ok()
            except Exception as e:
                span.set_error(str(e))
                span.add_event("exception", {
                    "exception.type": type(e).__name__,
                    "exception.message": str(e),
                })
                raise
            finally:
                span.end()
                self._record_span(span)

    async def start_span_async(
        self,
        name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: dict[str, Any] | None = None,
    ):
        """Start a new span for async context."""
        # Same as sync but returns async context manager
        return self.start_span(name, kind, attributes)

    def _record_span(self, span: Span) -> None:
        """Record a completed span for export."""
        self._pending_spans.append(span)

        # Auto-flush if batch size reached
        if len(self._pending_spans) >= self._batch_size:
            asyncio.create_task(self.flush())

    async def flush(self) -> None:
        """Flush pending spans to exporters."""
        if not self._pending_spans:
            return

        spans_to_export = self._pending_spans.copy()
        self._pending_spans.clear()

        for exporter in self.exporters:
            try:
                await exporter.export(spans_to_export)
            except Exception as e:
                # Log but don't fail
                print(f"Failed to export spans to {exporter.__class__.__name__}: {e}")

    async def shutdown(self) -> None:
        """Shutdown the tracer and exporters."""
        await self.flush()
        for exporter in self.exporters:
            await exporter.shutdown()


# Global tracer instance
_tracer: Tracer | None = None


def get_tracer() -> Tracer:
    """Get the global tracer instance."""
    global _tracer
    if _tracer is None:
        _tracer = Tracer()
    return _tracer


def set_tracer(tracer: Tracer) -> None:
    """Set the global tracer instance."""
    global _tracer
    _tracer = tracer


def trace(
    name: str | None = None,
    kind: SpanKind = SpanKind.INTERNAL,
    attributes: dict[str, Any] | None = None,
) -> Callable[[F], F]:
    """Decorator for tracing functions."""

    def decorator(func: F) -> F:
        span_name = name or func.__name__

        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                tracer = get_tracer()
                with tracer.start_span(span_name, kind, attributes) as span:
                    return await func(*args, **kwargs)
            return async_wrapper  # type: ignore
        else:
            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                tracer = get_tracer()
                with tracer.start_span(span_name, kind, attributes) as span:
                    return func(*args, **kwargs)
            return sync_wrapper  # type: ignore

    return decorator
