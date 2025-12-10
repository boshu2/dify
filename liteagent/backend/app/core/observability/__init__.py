"""
Observability module for LiteAgent.

Provides tracing, metrics, and logging infrastructure compatible with:
- OpenTelemetry (OTLP export)
- LangFuse
- LangSmith
- Arize Phoenix
- Opik
- Weave (Weights & Biases)
- Sentry
"""

from app.core.observability.config import ObservabilityConfig, get_observability_config
from app.core.observability.context import (
    TraceContext,
    get_current_trace_id,
    get_current_span_id,
    set_trace_context,
)
from app.core.observability.tracer import (
    Tracer,
    Span,
    SpanKind,
    get_tracer,
    trace,
)

__all__ = [
    "ObservabilityConfig",
    "get_observability_config",
    "TraceContext",
    "get_current_trace_id",
    "get_current_span_id",
    "set_trace_context",
    "Tracer",
    "Span",
    "SpanKind",
    "get_tracer",
    "trace",
]
