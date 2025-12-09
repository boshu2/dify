"""
Arize Phoenix exporter for ML observability.

Phoenix provides local-first tracing and evaluation for LLM applications.
"""

from typing import Any

import httpx

from app.core.observability.tracer import Span, SpanExporter


class ArizePhoenixExporter(SpanExporter):
    """Export spans to Arize Phoenix."""

    def __init__(
        self,
        endpoint: str = "http://localhost:6006",
        project_name: str = "default",
    ):
        self.endpoint = endpoint.rstrip("/")
        self.project_name = project_name
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.endpoint,
                headers={"Content-Type": "application/json"},
                timeout=30.0,
            )
        return self._client

    def _span_to_phoenix_span(self, span: Span) -> dict[str, Any]:
        """Convert span to Phoenix OTLP-like format."""
        phoenix_span: dict[str, Any] = {
            "context": {
                "trace_id": span.trace_id,
                "span_id": span.span_id,
            },
            "name": span.name,
            "kind": self._map_span_kind(span.kind.value),
            "start_time": span.start_time.isoformat(),
            "status": {
                "status_code": "OK" if span.status.value == "ok" else "ERROR",
                "message": span.status_message,
            },
            "attributes": self._build_attributes(span),
            "events": [
                {
                    "name": e.name,
                    "timestamp": e.timestamp.isoformat(),
                    "attributes": e.attributes,
                }
                for e in span.events
            ],
        }

        if span.parent_span_id:
            phoenix_span["parent_id"] = span.parent_span_id

        if span.end_time:
            phoenix_span["end_time"] = span.end_time.isoformat()

        return phoenix_span

    def _map_span_kind(self, kind: str) -> int:
        """Map span kind to OTLP kind."""
        mapping = {
            "internal": 1,
            "server": 2,
            "client": 3,
            "producer": 4,
            "consumer": 5,
        }
        return mapping.get(kind, 1)

    def _build_attributes(self, span: Span) -> dict[str, Any]:
        """Build Phoenix-compatible attributes."""
        attrs: dict[str, Any] = {
            "project": self.project_name,
            **span.attributes,
        }

        # Add LLM-specific attributes
        if span.model:
            attrs["llm.model_name"] = span.model
        if span.provider:
            attrs["llm.provider"] = span.provider
        if span.prompt_tokens is not None:
            attrs["llm.token_count.prompt"] = span.prompt_tokens
        if span.completion_tokens is not None:
            attrs["llm.token_count.completion"] = span.completion_tokens
        if span.total_tokens is not None:
            attrs["llm.token_count.total"] = span.total_tokens

        # Add tool-specific attributes
        if span.tool_name:
            attrs["tool.name"] = span.tool_name
        if span.tool_input:
            attrs["tool.input"] = span.tool_input
        if span.tool_output:
            attrs["tool.output"] = span.tool_output

        return attrs

    async def export(self, spans: list[Span]) -> bool:
        """Export spans to Phoenix."""
        if not spans:
            return True

        phoenix_spans = [self._span_to_phoenix_span(span) for span in spans]

        try:
            client = await self._get_client()
            response = await client.post(
                "/v1/traces",
                json={"spans": phoenix_spans},
            )
            response.raise_for_status()
            return True
        except Exception as e:
            print(f"Failed to export to Arize Phoenix: {e}")
            return False

    async def shutdown(self) -> None:
        """Shutdown the exporter."""
        if self._client:
            await self._client.aclose()
            self._client = None
