"""
LangSmith exporter for LangChain ecosystem tracing.

LangSmith provides tracing, evaluation, and monitoring for LLM applications.
"""

import asyncio
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import httpx

from app.core.observability.tracer import Span, SpanExporter


class LangSmithExporter(SpanExporter):
    """Export spans to LangSmith."""

    def __init__(
        self,
        api_key: str,
        project_name: str = "default",
        endpoint: str = "https://api.smith.langchain.com",
    ):
        self.api_key = api_key
        self.project_name = project_name
        self.endpoint = endpoint.rstrip("/")
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.endpoint,
                headers={
                    "x-api-key": self.api_key,
                    "Content-Type": "application/json",
                },
                timeout=30.0,
            )
        return self._client

    def _span_to_run(self, span: Span) -> dict[str, Any]:
        """Convert span to LangSmith run format."""
        run_type = self._determine_run_type(span)

        run: dict[str, Any] = {
            "id": span.span_id,
            "name": span.name,
            "run_type": run_type,
            "start_time": span.start_time.isoformat(),
            "inputs": self._get_inputs(span),
            "session_name": self.project_name,
            "trace_id": span.trace_id,
        }

        if span.parent_span_id:
            run["parent_run_id"] = span.parent_span_id

        if span.end_time:
            run["end_time"] = span.end_time.isoformat()
            run["outputs"] = self._get_outputs(span)

        if span.status.value == "error":
            run["error"] = span.status_message or "Unknown error"

        # Add extra metadata
        run["extra"] = {
            "metadata": span.attributes,
            "runtime": {
                "library": "liteagent",
            },
        }

        # Add token usage for LLM runs
        if run_type == "llm" and any([
            span.prompt_tokens,
            span.completion_tokens,
            span.total_tokens,
        ]):
            run["extra"]["token_usage"] = {
                "prompt_tokens": span.prompt_tokens,
                "completion_tokens": span.completion_tokens,
                "total_tokens": span.total_tokens,
            }

        return run

    def _determine_run_type(self, span: Span) -> str:
        """Determine LangSmith run type from span."""
        if span.model or span.provider:
            return "llm"
        elif span.tool_name:
            return "tool"
        elif "chain" in span.name.lower():
            return "chain"
        elif "retriever" in span.name.lower() or "retrieval" in span.name.lower():
            return "retriever"
        elif "agent" in span.name.lower():
            return "chain"  # Agents are chains in LangSmith
        else:
            return "chain"

    def _get_inputs(self, span: Span) -> dict[str, Any]:
        """Extract inputs from span."""
        inputs: dict[str, Any] = {}

        if span.tool_input:
            inputs["tool_input"] = span.tool_input
        elif "input" in span.attributes:
            inputs["input"] = span.attributes["input"]
        elif "messages" in span.attributes:
            inputs["messages"] = span.attributes["messages"]
        elif "query" in span.attributes:
            inputs["query"] = span.attributes["query"]

        return inputs

    def _get_outputs(self, span: Span) -> dict[str, Any]:
        """Extract outputs from span."""
        outputs: dict[str, Any] = {}

        if span.tool_output is not None:
            outputs["tool_output"] = span.tool_output
        elif "output" in span.attributes:
            outputs["output"] = span.attributes["output"]
        elif "response" in span.attributes:
            outputs["response"] = span.attributes["response"]

        return outputs

    async def export(self, spans: list[Span]) -> bool:
        """Export spans to LangSmith."""
        if not spans:
            return True

        runs = [self._span_to_run(span) for span in spans]

        try:
            client = await self._get_client()

            # LangSmith expects runs to be posted individually or in batches
            for run in runs:
                response = await client.post("/runs", json=run)
                response.raise_for_status()

            return True
        except Exception as e:
            print(f"Failed to export to LangSmith: {e}")
            return False

    async def shutdown(self) -> None:
        """Shutdown the exporter."""
        if self._client:
            await self._client.aclose()
            self._client = None


class LangSmithCallback:
    """Callback handler for LangSmith tracing."""

    def __init__(self, exporter: LangSmithExporter):
        self.exporter = exporter
        self._runs: dict[str, dict[str, Any]] = {}

    async def on_chain_start(
        self,
        trace_id: str,
        run_id: str,
        parent_run_id: str | None,
        name: str,
        inputs: dict[str, Any],
        **kwargs: Any,
    ) -> None:
        """Called when a chain starts."""
        run = {
            "id": run_id,
            "name": name,
            "run_type": "chain",
            "start_time": datetime.now(timezone.utc).isoformat(),
            "inputs": inputs,
            "session_name": self.exporter.project_name,
            "trace_id": trace_id,
            "extra": {"metadata": kwargs},
        }
        if parent_run_id:
            run["parent_run_id"] = parent_run_id
        self._runs[run_id] = run

    async def on_chain_end(
        self,
        run_id: str,
        outputs: dict[str, Any],
        **kwargs: Any,
    ) -> None:
        """Called when a chain ends."""
        run = self._runs.pop(run_id, None)
        if run:
            run["end_time"] = datetime.now(timezone.utc).isoformat()
            run["outputs"] = outputs
            await self._post_run(run)

    async def on_chain_error(
        self,
        run_id: str,
        error: str,
        **kwargs: Any,
    ) -> None:
        """Called when a chain errors."""
        run = self._runs.pop(run_id, None)
        if run:
            run["end_time"] = datetime.now(timezone.utc).isoformat()
            run["error"] = error
            await self._post_run(run)

    async def _post_run(self, run: dict[str, Any]) -> None:
        """Post a run to LangSmith."""
        try:
            client = await self.exporter._get_client()
            await client.post("/runs", json=run)
        except Exception as e:
            print(f"Failed to post run to LangSmith: {e}")
