"""
LangFuse exporter for LLM observability.

LangFuse provides specialized tracing for LLM applications with
prompt management, evaluation, and cost tracking.
"""

import asyncio
from typing import Any

import httpx

from app.core.observability.tracer import Span, SpanExporter, SpanKind


class LangFuseExporter(SpanExporter):
    """Export spans to LangFuse."""

    def __init__(
        self,
        public_key: str,
        secret_key: str,
        host: str = "https://cloud.langfuse.com",
        batch_size: int = 50,
        flush_interval: float = 5.0,
    ):
        self.public_key = public_key
        self.secret_key = secret_key
        self.host = host.rstrip("/")
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self._client: httpx.AsyncClient | None = None
        self._pending_events: list[dict[str, Any]] = []

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.host,
                auth=(self.public_key, self.secret_key),
                headers={"Content-Type": "application/json"},
                timeout=30.0,
            )
        return self._client

    def _span_to_langfuse_event(self, span: Span) -> dict[str, Any]:
        """Convert span to LangFuse event format."""
        # Determine event type based on span attributes
        if span.model or span.provider:
            return self._create_generation_event(span)
        elif span.tool_name:
            return self._create_span_event(span, "tool")
        elif span.kind == SpanKind.SERVER:
            return self._create_trace_event(span)
        else:
            return self._create_span_event(span)

    def _create_trace_event(self, span: Span) -> dict[str, Any]:
        """Create a trace event for top-level spans."""
        return {
            "type": "trace-create",
            "body": {
                "id": span.trace_id,
                "name": span.name,
                "input": span.attributes.get("input"),
                "output": span.attributes.get("output"),
                "metadata": {
                    k: v for k, v in span.attributes.items()
                    if k not in ("input", "output")
                },
                "tags": span.attributes.get("tags", []),
                "userId": span.attributes.get("user_id"),
                "sessionId": span.attributes.get("session_id"),
            },
        }

    def _create_generation_event(self, span: Span) -> dict[str, Any]:
        """Create a generation event for LLM calls."""
        usage = {}
        if span.prompt_tokens is not None:
            usage["promptTokens"] = span.prompt_tokens
        if span.completion_tokens is not None:
            usage["completionTokens"] = span.completion_tokens
        if span.total_tokens is not None:
            usage["totalTokens"] = span.total_tokens

        return {
            "type": "generation-create",
            "body": {
                "id": span.span_id,
                "traceId": span.trace_id,
                "parentObservationId": span.parent_span_id,
                "name": span.name,
                "model": span.model,
                "modelParameters": span.attributes.get("model_parameters", {}),
                "input": span.attributes.get("input"),
                "output": span.attributes.get("output"),
                "usage": usage if usage else None,
                "startTime": span.start_time.isoformat(),
                "endTime": span.end_time.isoformat() if span.end_time else None,
                "completionStartTime": span.attributes.get("first_token_time"),
                "metadata": {
                    "provider": span.provider,
                    "status": span.status.value,
                    **{k: v for k, v in span.attributes.items()
                       if k not in ("input", "output", "model_parameters", "first_token_time")},
                },
                "level": "ERROR" if span.status.value == "error" else "DEFAULT",
                "statusMessage": span.status_message,
            },
        }

    def _create_span_event(
        self,
        span: Span,
        span_type: str = "span",
    ) -> dict[str, Any]:
        """Create a span event."""
        return {
            "type": "span-create",
            "body": {
                "id": span.span_id,
                "traceId": span.trace_id,
                "parentObservationId": span.parent_span_id,
                "name": span.name,
                "input": span.tool_input or span.attributes.get("input"),
                "output": span.tool_output or span.attributes.get("output"),
                "startTime": span.start_time.isoformat(),
                "endTime": span.end_time.isoformat() if span.end_time else None,
                "metadata": {
                    "span_type": span_type,
                    "tool_name": span.tool_name,
                    "status": span.status.value,
                    **span.attributes,
                },
                "level": "ERROR" if span.status.value == "error" else "DEFAULT",
                "statusMessage": span.status_message,
            },
        }

    async def export(self, spans: list[Span]) -> bool:
        """Export spans to LangFuse."""
        if not spans:
            return True

        events = [self._span_to_langfuse_event(span) for span in spans]

        try:
            client = await self._get_client()
            response = await client.post(
                "/api/public/ingestion",
                json={"batch": events},
            )
            response.raise_for_status()
            return True
        except Exception as e:
            print(f"Failed to export to LangFuse: {e}")
            return False

    async def shutdown(self) -> None:
        """Shutdown the exporter."""
        if self._client:
            await self._client.aclose()
            self._client = None


class LangFuseCallback:
    """Callback handler for LangFuse tracing in agent/LLM calls."""

    def __init__(self, exporter: LangFuseExporter):
        self.exporter = exporter
        self._spans: dict[str, Span] = {}

    async def on_llm_start(
        self,
        trace_id: str,
        span_id: str,
        parent_span_id: str | None,
        model: str,
        provider: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> None:
        """Called when LLM call starts."""
        span = Span(
            name=f"llm.{provider}.{model}",
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            model=model,
            provider=provider,
            attributes={
                "input": messages,
                **kwargs,
            },
        )
        self._spans[span_id] = span

    async def on_llm_end(
        self,
        span_id: str,
        output: str,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        total_tokens: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when LLM call ends."""
        span = self._spans.pop(span_id, None)
        if span:
            span.end()
            span.set_ok()
            span.set_attribute("output", output)
            span.prompt_tokens = prompt_tokens
            span.completion_tokens = completion_tokens
            span.total_tokens = total_tokens
            await self.exporter.export([span])

    async def on_llm_error(
        self,
        span_id: str,
        error: str,
        **kwargs: Any,
    ) -> None:
        """Called when LLM call fails."""
        span = self._spans.pop(span_id, None)
        if span:
            span.end()
            span.set_error(error)
            await self.exporter.export([span])

    async def on_tool_start(
        self,
        trace_id: str,
        span_id: str,
        parent_span_id: str | None,
        tool_name: str,
        tool_input: dict[str, Any],
        **kwargs: Any,
    ) -> None:
        """Called when tool execution starts."""
        span = Span(
            name=f"tool.{tool_name}",
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            tool_name=tool_name,
            tool_input=tool_input,
        )
        self._spans[span_id] = span

    async def on_tool_end(
        self,
        span_id: str,
        output: Any,
        **kwargs: Any,
    ) -> None:
        """Called when tool execution ends."""
        span = self._spans.pop(span_id, None)
        if span:
            span.end()
            span.set_ok()
            span.tool_output = output
            await self.exporter.export([span])

    async def on_tool_error(
        self,
        span_id: str,
        error: str,
        **kwargs: Any,
    ) -> None:
        """Called when tool execution fails."""
        span = self._spans.pop(span_id, None)
        if span:
            span.end()
            span.set_error(error)
            await self.exporter.export([span])
