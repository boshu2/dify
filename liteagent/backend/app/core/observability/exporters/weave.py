"""
Weave exporter for Weights & Biases LLM observability.

Weave provides tracing and evaluation integrated with W&B ecosystem.
"""

from typing import Any

import httpx

from app.core.observability.tracer import Span, SpanExporter


class WeaveExporter(SpanExporter):
    """Export spans to Weave (Weights & Biases)."""

    def __init__(
        self,
        api_key: str,
        project_name: str = "liteagent",
        entity: str = "",
        endpoint: str = "https://api.wandb.ai",
    ):
        self.api_key = api_key
        self.project_name = project_name
        self.entity = entity
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

    def _span_to_weave_call(self, span: Span) -> dict[str, Any]:
        """Convert span to Weave call format."""
        call: dict[str, Any] = {
            "id": span.span_id,
            "trace_id": span.trace_id,
            "op_name": span.name,
            "project_id": f"{self.entity}/{self.project_name}" if self.entity else self.project_name,
            "started_at": span.start_time.isoformat(),
            "attributes": span.attributes,
        }

        if span.parent_span_id:
            call["parent_id"] = span.parent_span_id

        if span.end_time:
            call["ended_at"] = span.end_time.isoformat()

        # Add inputs
        inputs: dict[str, Any] = {}
        if "input" in span.attributes:
            inputs["input"] = span.attributes["input"]
        if "messages" in span.attributes:
            inputs["messages"] = span.attributes["messages"]
        if span.tool_input:
            inputs["tool_input"] = span.tool_input
        if inputs:
            call["inputs"] = inputs

        # Add outputs
        outputs: dict[str, Any] = {}
        if "output" in span.attributes:
            outputs["output"] = span.attributes["output"]
        if span.tool_output:
            outputs["tool_output"] = span.tool_output
        if outputs:
            call["outputs"] = outputs

        # Add LLM-specific data
        if span.model:
            call["model"] = span.model
        if span.provider:
            call["provider"] = span.provider

        # Add summary with usage
        summary: dict[str, Any] = {}
        if span.prompt_tokens is not None:
            summary["prompt_tokens"] = span.prompt_tokens
        if span.completion_tokens is not None:
            summary["completion_tokens"] = span.completion_tokens
        if span.total_tokens is not None:
            summary["total_tokens"] = span.total_tokens
        if span.duration_ms is not None:
            summary["latency_ms"] = span.duration_ms
        if summary:
            call["summary"] = summary

        # Add error info
        if span.status.value == "error":
            call["exception"] = span.status_message

        return call

    async def export(self, spans: list[Span]) -> bool:
        """Export spans to Weave."""
        if not spans:
            return True

        calls = [self._span_to_weave_call(span) for span in spans]

        try:
            client = await self._get_client()
            # Weave uses GraphQL API
            response = await client.post(
                "/graphql",
                json={
                    "query": """
                        mutation CreateCalls($calls: [CallInput!]!) {
                            createCalls(calls: $calls) {
                                success
                            }
                        }
                    """,
                    "variables": {"calls": calls},
                },
            )
            response.raise_for_status()
            return True
        except Exception as e:
            print(f"Failed to export to Weave: {e}")
            return False

    async def shutdown(self) -> None:
        """Shutdown the exporter."""
        if self._client:
            await self._client.aclose()
            self._client = None
