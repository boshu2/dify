"""
Opik exporter for Comet ML LLM observability.

Opik provides tracing, evaluation, and prompt management.
"""

from typing import Any

import httpx

from app.core.observability.tracer import Span, SpanExporter


class OpikExporter(SpanExporter):
    """Export spans to Opik (Comet ML)."""

    def __init__(
        self,
        api_key: str,
        workspace: str = "",
        project_name: str = "default",
        endpoint: str = "https://www.comet.com/opik/api",
    ):
        self.api_key = api_key
        self.workspace = workspace
        self.project_name = project_name
        self.endpoint = endpoint.rstrip("/")
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.endpoint,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=30.0,
            )
        return self._client

    def _span_to_opik_trace(self, span: Span) -> dict[str, Any]:
        """Convert span to Opik trace format."""
        trace: dict[str, Any] = {
            "id": span.span_id,
            "trace_id": span.trace_id,
            "name": span.name,
            "project_name": self.project_name,
            "start_time": span.start_time.isoformat(),
            "metadata": span.attributes,
        }

        if span.parent_span_id:
            trace["parent_span_id"] = span.parent_span_id

        if span.end_time:
            trace["end_time"] = span.end_time.isoformat()
            trace["duration_ms"] = span.duration_ms

        # Add type-specific data
        if span.model or span.provider:
            trace["type"] = "llm"
            trace["model"] = span.model
            trace["provider"] = span.provider
            if span.prompt_tokens:
                trace["usage"] = {
                    "prompt_tokens": span.prompt_tokens,
                    "completion_tokens": span.completion_tokens,
                    "total_tokens": span.total_tokens,
                }
        elif span.tool_name:
            trace["type"] = "tool"
            trace["tool_name"] = span.tool_name
            trace["tool_input"] = span.tool_input
            trace["tool_output"] = span.tool_output
        else:
            trace["type"] = "general"

        # Add input/output
        if "input" in span.attributes:
            trace["input"] = span.attributes["input"]
        if "output" in span.attributes:
            trace["output"] = span.attributes["output"]

        # Add error info
        if span.status.value == "error":
            trace["error"] = True
            trace["error_message"] = span.status_message

        return trace

    async def export(self, spans: list[Span]) -> bool:
        """Export spans to Opik."""
        if not spans:
            return True

        traces = [self._span_to_opik_trace(span) for span in spans]

        try:
            client = await self._get_client()
            response = await client.post(
                "/v1/private/traces/batch",
                json={"traces": traces},
            )
            response.raise_for_status()
            return True
        except Exception as e:
            print(f"Failed to export to Opik: {e}")
            return False

    async def shutdown(self) -> None:
        """Shutdown the exporter."""
        if self._client:
            await self._client.aclose()
            self._client = None
