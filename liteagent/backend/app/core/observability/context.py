"""
Trace context management for request correlation.

Provides context propagation for distributed tracing across async operations.
"""

import secrets
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class TraceContext:
    """Context for distributed tracing."""

    trace_id: str = field(default_factory=lambda: secrets.token_hex(16))
    span_id: str = field(default_factory=lambda: secrets.token_hex(8))
    parent_span_id: str | None = None
    baggage: dict[str, str] = field(default_factory=dict)
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Additional context
    user_id: str | None = None
    tenant_id: str | None = None
    request_id: str | None = None
    session_id: str | None = None
    agent_id: str | None = None
    conversation_id: str | None = None
    workflow_id: str | None = None

    def create_child(self) -> "TraceContext":
        """Create a child span context."""
        return TraceContext(
            trace_id=self.trace_id,
            span_id=secrets.token_hex(8),
            parent_span_id=self.span_id,
            baggage=self.baggage.copy(),
            user_id=self.user_id,
            tenant_id=self.tenant_id,
            request_id=self.request_id,
            session_id=self.session_id,
            agent_id=self.agent_id,
            conversation_id=self.conversation_id,
            workflow_id=self.workflow_id,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "baggage": self.baggage,
            "start_time": self.start_time.isoformat(),
            "user_id": self.user_id,
            "tenant_id": self.tenant_id,
            "request_id": self.request_id,
            "session_id": self.session_id,
            "agent_id": self.agent_id,
            "conversation_id": self.conversation_id,
            "workflow_id": self.workflow_id,
        }

    def to_headers(self) -> dict[str, str]:
        """Convert to W3C Trace Context headers."""
        headers = {
            "traceparent": f"00-{self.trace_id}-{self.span_id}-01",
        }
        if self.baggage:
            baggage_str = ",".join(f"{k}={v}" for k, v in self.baggage.items())
            headers["baggage"] = baggage_str
        return headers

    @classmethod
    def from_headers(cls, headers: dict[str, str]) -> "TraceContext":
        """Parse from W3C Trace Context headers."""
        traceparent = headers.get("traceparent", "")
        baggage_str = headers.get("baggage", "")

        trace_id = secrets.token_hex(16)
        parent_span_id = None

        if traceparent:
            parts = traceparent.split("-")
            if len(parts) >= 3:
                trace_id = parts[1]
                parent_span_id = parts[2]

        baggage = {}
        if baggage_str:
            for item in baggage_str.split(","):
                if "=" in item:
                    key, value = item.split("=", 1)
                    baggage[key.strip()] = value.strip()

        return cls(
            trace_id=trace_id,
            parent_span_id=parent_span_id,
            baggage=baggage,
        )


# Context variable for the current trace
_current_context: ContextVar[TraceContext | None] = ContextVar(
    "trace_context",
    default=None,
)


def get_current_context() -> TraceContext | None:
    """Get the current trace context."""
    return _current_context.get()


def get_current_trace_id() -> str | None:
    """Get the current trace ID."""
    ctx = _current_context.get()
    return ctx.trace_id if ctx else None


def get_current_span_id() -> str | None:
    """Get the current span ID."""
    ctx = _current_context.get()
    return ctx.span_id if ctx else None


def get_current_request_id() -> str | None:
    """Get the current request ID."""
    ctx = _current_context.get()
    return ctx.request_id if ctx else None


def set_trace_context(context: TraceContext | None) -> None:
    """Set the current trace context."""
    _current_context.set(context)


class trace_context:
    """Context manager for trace context."""

    def __init__(
        self,
        context: TraceContext | None = None,
        **kwargs: Any,
    ):
        """Initialize with optional context or kwargs to create one."""
        if context:
            self.context = context
        else:
            # Check for parent context
            parent = _current_context.get()
            if parent:
                self.context = parent.create_child()
                # Apply any overrides
                for key, value in kwargs.items():
                    if hasattr(self.context, key):
                        setattr(self.context, key, value)
            else:
                self.context = TraceContext(**kwargs)
        self.token = None

    def __enter__(self) -> TraceContext:
        self.token = _current_context.set(self.context)
        return self.context

    def __exit__(self, *args: Any) -> None:
        if self.token is not None:
            _current_context.reset(self.token)


def generate_request_id() -> str:
    """Generate a unique request ID."""
    return str(uuid.uuid4())


def generate_trace_id() -> str:
    """Generate a unique trace ID (32 hex chars)."""
    return secrets.token_hex(16)


def generate_span_id() -> str:
    """Generate a unique span ID (16 hex chars)."""
    return secrets.token_hex(8)
