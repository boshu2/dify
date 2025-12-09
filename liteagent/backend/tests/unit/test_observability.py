"""
Tests for observability features.
"""

import asyncio
import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.observability.config import (
    ObservabilityConfig,
    OTLPProtocol,
    TracingProvider,
)
from app.core.observability.context import (
    TraceContext,
    generate_request_id,
    generate_span_id,
    generate_trace_id,
    get_current_context,
    get_current_span_id,
    get_current_trace_id,
    set_trace_context,
    trace_context,
)
from app.core.observability.tracer import (
    ConsoleExporter,
    InMemoryExporter,
    Span,
    SpanEvent,
    SpanKind,
    SpanStatus,
    Tracer,
    get_tracer,
    set_tracer,
    trace,
)


class TestObservabilityConfig:
    """Tests for ObservabilityConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ObservabilityConfig()

        assert config.enabled is True
        assert config.service_name == "liteagent"
        assert config.tracing_provider == TracingProvider.NONE
        assert config.otel_enabled is False
        assert config.langfuse_enabled is False
        assert config.sentry_enabled is False

    def test_config_tracing_providers(self):
        """Test tracing provider enum values."""
        assert TracingProvider.LANGFUSE.value == "langfuse"
        assert TracingProvider.LANGSMITH.value == "langsmith"
        assert TracingProvider.ARIZE_PHOENIX.value == "arize_phoenix"
        assert TracingProvider.OPIK.value == "opik"
        assert TracingProvider.WEAVE.value == "weave"
        assert TracingProvider.OTEL.value == "otel"

    def test_config_otlp_protocols(self):
        """Test OTLP protocol enum values."""
        assert OTLPProtocol.GRPC.value == "grpc"
        assert OTLPProtocol.HTTP.value == "http/protobuf"


class TestTraceContext:
    """Tests for TraceContext."""

    def test_create_context(self):
        """Test creating a trace context."""
        ctx = TraceContext()

        assert ctx.trace_id is not None
        assert len(ctx.trace_id) == 32  # 16 bytes = 32 hex chars
        assert ctx.span_id is not None
        assert len(ctx.span_id) == 16  # 8 bytes = 16 hex chars
        assert ctx.parent_span_id is None
        assert ctx.baggage == {}

    def test_create_child_context(self):
        """Test creating a child span context."""
        parent = TraceContext()
        child = parent.create_child()

        assert child.trace_id == parent.trace_id
        assert child.span_id != parent.span_id
        assert child.parent_span_id == parent.span_id

    def test_context_to_dict(self):
        """Test converting context to dictionary."""
        ctx = TraceContext(
            user_id="user-123",
            agent_id="agent-456",
        )
        data = ctx.to_dict()

        assert data["trace_id"] == ctx.trace_id
        assert data["span_id"] == ctx.span_id
        assert data["user_id"] == "user-123"
        assert data["agent_id"] == "agent-456"

    def test_context_to_headers(self):
        """Test W3C Trace Context header generation."""
        ctx = TraceContext()
        headers = ctx.to_headers()

        assert "traceparent" in headers
        assert headers["traceparent"].startswith("00-")
        assert ctx.trace_id in headers["traceparent"]
        assert ctx.span_id in headers["traceparent"]

    def test_context_from_headers(self):
        """Test parsing W3C Trace Context headers."""
        headers = {
            "traceparent": "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01",
            "baggage": "key1=value1,key2=value2",
        }
        ctx = TraceContext.from_headers(headers)

        assert ctx.trace_id == "0af7651916cd43dd8448eb211c80319c"
        assert ctx.parent_span_id == "b7ad6b7169203331"
        assert ctx.baggage["key1"] == "value1"
        assert ctx.baggage["key2"] == "value2"

    def test_trace_context_manager(self):
        """Test trace context manager."""
        with trace_context(user_id="user-123") as ctx:
            current = get_current_context()
            assert current is not None
            assert current.user_id == "user-123"

        # Context should be cleared after exit
        assert get_current_context() is None

    def test_nested_trace_contexts(self):
        """Test nested trace contexts."""
        with trace_context() as parent:
            assert get_current_trace_id() == parent.trace_id

            with trace_context() as child:
                assert get_current_trace_id() == parent.trace_id  # Same trace
                assert get_current_span_id() == child.span_id
                assert child.parent_span_id == parent.span_id

            # Back to parent
            assert get_current_span_id() == parent.span_id


class TestSpan:
    """Tests for Span."""

    def test_create_span(self):
        """Test creating a span."""
        span = Span(
            name="test-span",
            trace_id="abc123",
            span_id="def456",
        )

        assert span.name == "test-span"
        assert span.trace_id == "abc123"
        assert span.span_id == "def456"
        assert span.kind == SpanKind.INTERNAL
        assert span.status == SpanStatus.UNSET

    def test_span_attributes(self):
        """Test setting span attributes."""
        span = Span(name="test", trace_id="a", span_id="b")

        span.set_attribute("key1", "value1")
        span.set_attributes({"key2": "value2", "key3": 123})

        assert span.attributes["key1"] == "value1"
        assert span.attributes["key2"] == "value2"
        assert span.attributes["key3"] == 123

    def test_span_events(self):
        """Test adding span events."""
        span = Span(name="test", trace_id="a", span_id="b")

        span.add_event("event1", {"data": "value"})
        span.add_event("event2")

        assert len(span.events) == 2
        assert span.events[0].name == "event1"
        assert span.events[0].attributes["data"] == "value"
        assert span.events[1].name == "event2"

    def test_span_status(self):
        """Test setting span status."""
        span = Span(name="test", trace_id="a", span_id="b")

        span.set_ok()
        assert span.status == SpanStatus.OK

        span.set_error("Something failed")
        assert span.status == SpanStatus.ERROR
        assert span.status_message == "Something failed"

    def test_span_duration(self):
        """Test span duration calculation."""
        span = Span(name="test", trace_id="a", span_id="b")
        span.end()

        assert span.end_time is not None
        assert span.duration_ms is not None
        assert span.duration_ms >= 0

    def test_span_to_dict(self):
        """Test span serialization."""
        span = Span(
            name="llm-call",
            trace_id="trace123",
            span_id="span456",
            model="gpt-4",
            provider="openai",
            prompt_tokens=100,
            completion_tokens=50,
        )
        span.set_ok()
        span.end()

        data = span.to_dict()

        assert data["name"] == "llm-call"
        assert data["trace_id"] == "trace123"
        assert data["model"] == "gpt-4"
        assert data["provider"] == "openai"
        assert data["prompt_tokens"] == 100
        assert data["completion_tokens"] == 50
        assert "duration_ms" in data


class TestTracer:
    """Tests for Tracer."""

    def test_create_tracer(self):
        """Test creating a tracer."""
        tracer = Tracer(service_name="test-service")

        assert tracer.service_name == "test-service"
        assert tracer.enabled is True
        assert tracer.exporters == []

    def test_tracer_with_exporter(self):
        """Test tracer with an exporter."""
        exporter = InMemoryExporter()
        tracer = Tracer(exporters=[exporter])

        with tracer.start_span("test-span") as span:
            span.set_attribute("key", "value")

        assert len(tracer._pending_spans) == 1

    def test_tracer_disabled(self):
        """Test disabled tracer."""
        tracer = Tracer(enabled=False)

        with tracer.start_span("test-span") as span:
            assert span is None

    def test_tracer_exception_handling(self):
        """Test tracer handles exceptions."""
        tracer = Tracer()

        with pytest.raises(ValueError):
            with tracer.start_span("test-span") as span:
                raise ValueError("Test error")

        # Span should still be recorded with error status
        assert len(tracer._pending_spans) == 1
        assert tracer._pending_spans[0].status == SpanStatus.ERROR

    @pytest.mark.asyncio
    async def test_tracer_flush(self):
        """Test flushing spans to exporters."""
        exporter = InMemoryExporter()
        tracer = Tracer(exporters=[exporter])

        with tracer.start_span("test-span"):
            pass

        await tracer.flush()

        assert len(exporter.spans) == 1
        assert len(tracer._pending_spans) == 0


class TestInMemoryExporter:
    """Tests for InMemoryExporter."""

    @pytest.mark.asyncio
    async def test_export_spans(self):
        """Test exporting spans."""
        exporter = InMemoryExporter()
        span = Span(name="test", trace_id="a", span_id="b")

        result = await exporter.export([span])

        assert result is True
        assert len(exporter.spans) == 1

    @pytest.mark.asyncio
    async def test_clear_spans(self):
        """Test clearing exported spans."""
        exporter = InMemoryExporter()
        span = Span(name="test", trace_id="a", span_id="b")
        await exporter.export([span])

        exporter.clear()

        assert len(exporter.spans) == 0


class TestTraceDecorator:
    """Tests for @trace decorator."""

    def test_sync_function_trace(self):
        """Test tracing a synchronous function."""
        tracer = Tracer()
        set_tracer(tracer)

        @trace("my-function")
        def my_function(x: int) -> int:
            return x * 2

        result = my_function(5)

        assert result == 10
        assert len(tracer._pending_spans) == 1
        assert tracer._pending_spans[0].name == "my-function"

    @pytest.mark.asyncio
    async def test_async_function_trace(self):
        """Test tracing an asynchronous function."""
        tracer = Tracer()
        set_tracer(tracer)

        @trace("async-function")
        async def async_function(x: int) -> int:
            return x * 2

        result = await async_function(5)

        assert result == 10
        assert len(tracer._pending_spans) == 1
        assert tracer._pending_spans[0].name == "async-function"


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_generate_request_id(self):
        """Test generating request IDs."""
        id1 = generate_request_id()
        id2 = generate_request_id()

        assert id1 != id2
        assert len(id1) == 36  # UUID format

    def test_generate_trace_id(self):
        """Test generating trace IDs."""
        trace_id = generate_trace_id()

        assert len(trace_id) == 32  # 16 bytes hex

    def test_generate_span_id(self):
        """Test generating span IDs."""
        span_id = generate_span_id()

        assert len(span_id) == 16  # 8 bytes hex


class TestLogging:
    """Tests for logging enhancements."""

    def test_log_config_defaults(self):
        """Test LogConfig default values."""
        from app.core.logging import LogConfig

        config = LogConfig()

        assert config.level == "INFO"
        assert config.format == "json"
        assert config.file_enabled is False

    def test_log_config_validation(self):
        """Test LogConfig validation."""
        from app.core.logging import LogConfig

        with pytest.raises(ValueError):
            LogConfig(level="INVALID")

    def test_json_formatter(self):
        """Test JSON log formatter."""
        import logging
        from io import StringIO

        from app.core.logging import JSONFormatter, setup_logging, LogConfig

        stream = StringIO()
        config = LogConfig(format="json")
        setup_logging(config, stream=stream)

        logger = logging.getLogger("liteagent.test")
        logger.info("Test message")

        output = stream.getvalue()
        log_data = json.loads(output.strip())

        assert log_data["message"] == "Test message"
        assert log_data["level"] == "INFO"
        assert "timestamp" in log_data

    def test_request_id_context(self):
        """Test request ID context manager."""
        from app.core.logging import RequestIDContext, get_request_id

        with RequestIDContext("test-request-123") as request_id:
            assert request_id == "test-request-123"
            assert get_request_id() == "test-request-123"

        assert get_request_id() is None

    def test_request_logger(self):
        """Test RequestLogger."""
        from app.core.logging import RequestLogger

        logger = RequestLogger()

        # Should not raise
        logger.log_request_start("GET", "/api/test", "req-123")
        logger.log_request_end("GET", "/api/test", 200, 50.0, "req-123")
        logger.log_request_error("GET", "/api/test", "Error", "req-123")

    def test_llm_call_logger(self):
        """Test LLMCallLogger."""
        from app.core.logging import LLMCallLogger

        logger = LLMCallLogger()

        # Should not raise
        logger.log_call_start("openai", "gpt-4", agent_id="agent-123")
        logger.log_call_end(
            "openai", "gpt-4", 1000.0,
            prompt_tokens=100,
            completion_tokens=50,
        )
        logger.log_call_error("openai", "gpt-4", "API Error")

    def test_workflow_logger(self):
        """Test WorkflowLogger."""
        from app.core.logging import WorkflowLogger

        logger = WorkflowLogger()

        # Should not raise
        logger.log_workflow_start("wf-123", "Test Workflow")
        logger.log_workflow_end("wf-123", "Test Workflow", "completed", 5000.0)
        logger.log_node_execution("wf-123", "node-1", "llm", "completed", 1000.0)

    def test_metrics_collector(self):
        """Test MetricsCollector."""
        from app.core.logging import MetricsCollector

        metrics = MetricsCollector()

        metrics.increment("requests_total")
        metrics.increment("requests_total")
        metrics.increment("requests_total", labels={"status": "200"})

        assert metrics.get_counter("requests_total") == 2
        assert metrics.get_counter("requests_total", {"status": "200"}) == 1

        metrics.record_histogram("request_duration", 100.0)
        metrics.record_histogram("request_duration", 200.0)

        histogram = metrics.get_histogram("request_duration")
        assert histogram == [100.0, 200.0]

        metrics.reset()
        assert metrics.get_counter("requests_total") == 0


class TestSentry:
    """Tests for Sentry integration."""

    def test_sentry_manager_without_dsn(self):
        """Test SentryManager without DSN."""
        from app.core.observability.sentry import SentryManager

        manager = SentryManager(dsn="")
        result = manager.initialize()

        assert result is False

    def test_sentry_capture_without_init(self):
        """Test capture functions without initialization."""
        from app.core.observability.sentry import capture_exception, capture_message

        # Should not raise, just return None
        result = capture_exception(ValueError("test"))
        assert result is None

        result = capture_message("test message")
        assert result is None


class TestTraceService:
    """Tests for TraceService."""

    @pytest.mark.asyncio
    async def test_trace_provider_types(self):
        """Test trace provider type enum."""
        from app.services.trace_service import TraceProviderType

        assert TraceProviderType.LANGFUSE.value == "langfuse"
        assert TraceProviderType.LANGSMITH.value == "langsmith"
        assert TraceProviderType.OTEL.value == "otel"

    def test_trace_manager_create_exporter(self):
        """Test TraceManager exporter creation."""
        from app.services.trace_service import TraceManager, TraceProviderType

        manager = TraceManager()

        # Test creating LangFuse exporter
        exporter = manager._create_exporter(
            TraceProviderType.LANGFUSE,
            {
                "public_key": "pk-test",
                "secret_key": "sk-test",
            },
        )

        assert exporter is not None
