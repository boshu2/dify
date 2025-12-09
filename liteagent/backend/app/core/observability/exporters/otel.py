"""
OpenTelemetry OTLP exporter for standard observability.

Supports both gRPC and HTTP/protobuf protocols for exporting traces.
"""

import json
from typing import Any

import httpx

from app.core.observability.config import OTLPProtocol
from app.core.observability.tracer import Span, SpanExporter


class OTLPExporter(SpanExporter):
    """Export spans via OTLP protocol."""

    def __init__(
        self,
        endpoint: str = "http://localhost:4317",
        protocol: OTLPProtocol = OTLPProtocol.GRPC,
        headers: dict[str, str] | None = None,
        service_name: str = "liteagent",
        service_version: str = "1.0.0",
    ):
        self.endpoint = endpoint.rstrip("/")
        self.protocol = protocol
        self.headers = headers or {}
        self.service_name = service_name
        self.service_version = service_version
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            # Adjust endpoint for HTTP protocol
            if self.protocol == OTLPProtocol.HTTP:
                base_url = self.endpoint
                if not base_url.endswith("/v1/traces"):
                    base_url = f"{base_url}/v1/traces"
            else:
                base_url = self.endpoint

            self._client = httpx.AsyncClient(
                base_url=base_url if self.protocol == OTLPProtocol.GRPC else self.endpoint,
                headers={
                    "Content-Type": "application/json",
                    **self.headers,
                },
                timeout=30.0,
            )
        return self._client

    def _span_to_otlp(self, span: Span) -> dict[str, Any]:
        """Convert span to OTLP span format."""
        # Convert trace_id and span_id to bytes representation (hex string)
        otlp_span: dict[str, Any] = {
            "traceId": span.trace_id,
            "spanId": span.span_id,
            "name": span.name,
            "kind": self._map_span_kind(span.kind.value),
            "startTimeUnixNano": int(span.start_time.timestamp() * 1e9),
            "status": {
                "code": 1 if span.status.value == "ok" else 2 if span.status.value == "error" else 0,
            },
            "attributes": self._to_otlp_attributes(span.attributes),
        }

        if span.parent_span_id:
            otlp_span["parentSpanId"] = span.parent_span_id

        if span.end_time:
            otlp_span["endTimeUnixNano"] = int(span.end_time.timestamp() * 1e9)

        if span.status_message:
            otlp_span["status"]["message"] = span.status_message

        # Add events
        if span.events:
            otlp_span["events"] = [
                {
                    "name": e.name,
                    "timeUnixNano": int(e.timestamp.timestamp() * 1e9),
                    "attributes": self._to_otlp_attributes(e.attributes),
                }
                for e in span.events
            ]

        # Add LLM-specific attributes
        if span.model:
            otlp_span["attributes"].append({
                "key": "llm.model",
                "value": {"stringValue": span.model},
            })
        if span.provider:
            otlp_span["attributes"].append({
                "key": "llm.provider",
                "value": {"stringValue": span.provider},
            })
        if span.prompt_tokens is not None:
            otlp_span["attributes"].append({
                "key": "llm.usage.prompt_tokens",
                "value": {"intValue": str(span.prompt_tokens)},
            })
        if span.completion_tokens is not None:
            otlp_span["attributes"].append({
                "key": "llm.usage.completion_tokens",
                "value": {"intValue": str(span.completion_tokens)},
            })
        if span.total_tokens is not None:
            otlp_span["attributes"].append({
                "key": "llm.usage.total_tokens",
                "value": {"intValue": str(span.total_tokens)},
            })

        return otlp_span

    def _map_span_kind(self, kind: str) -> int:
        """Map span kind to OTLP span kind."""
        mapping = {
            "internal": 1,
            "server": 2,
            "client": 3,
            "producer": 4,
            "consumer": 5,
        }
        return mapping.get(kind, 1)

    def _to_otlp_attributes(self, attrs: dict[str, Any]) -> list[dict[str, Any]]:
        """Convert attributes to OTLP format."""
        result = []
        for key, value in attrs.items():
            attr = {"key": key}
            if isinstance(value, bool):
                attr["value"] = {"boolValue": value}
            elif isinstance(value, int):
                attr["value"] = {"intValue": str(value)}
            elif isinstance(value, float):
                attr["value"] = {"doubleValue": value}
            elif isinstance(value, str):
                attr["value"] = {"stringValue": value}
            elif isinstance(value, (list, tuple)):
                attr["value"] = {"arrayValue": {"values": [
                    self._to_otlp_value(v) for v in value
                ]}}
            elif isinstance(value, dict):
                attr["value"] = {"stringValue": json.dumps(value)}
            else:
                attr["value"] = {"stringValue": str(value)}
            result.append(attr)
        return result

    def _to_otlp_value(self, value: Any) -> dict[str, Any]:
        """Convert a single value to OTLP format."""
        if isinstance(value, bool):
            return {"boolValue": value}
        elif isinstance(value, int):
            return {"intValue": str(value)}
        elif isinstance(value, float):
            return {"doubleValue": value}
        elif isinstance(value, str):
            return {"stringValue": value}
        else:
            return {"stringValue": str(value)}

    def _build_resource(self) -> dict[str, Any]:
        """Build OTLP resource."""
        return {
            "attributes": [
                {
                    "key": "service.name",
                    "value": {"stringValue": self.service_name},
                },
                {
                    "key": "service.version",
                    "value": {"stringValue": self.service_version},
                },
            ],
        }

    async def export(self, spans: list[Span]) -> bool:
        """Export spans via OTLP."""
        if not spans:
            return True

        otlp_spans = [self._span_to_otlp(span) for span in spans]

        # Build OTLP request
        request_body = {
            "resourceSpans": [
                {
                    "resource": self._build_resource(),
                    "scopeSpans": [
                        {
                            "scope": {
                                "name": "liteagent",
                                "version": self.service_version,
                            },
                            "spans": otlp_spans,
                        }
                    ],
                }
            ]
        }

        try:
            client = await self._get_client()

            if self.protocol == OTLPProtocol.HTTP:
                response = await client.post(
                    "/v1/traces",
                    json=request_body,
                )
            else:
                # For gRPC, we'd typically use grpcio
                # This is a fallback HTTP implementation
                response = await client.post(
                    "/v1/traces",
                    json=request_body,
                )

            response.raise_for_status()
            return True
        except Exception as e:
            print(f"Failed to export via OTLP: {e}")
            return False

    async def shutdown(self) -> None:
        """Shutdown the exporter."""
        if self._client:
            await self._client.aclose()
            self._client = None


class OTLPMetricsExporter:
    """Export metrics via OTLP protocol."""

    def __init__(
        self,
        endpoint: str = "http://localhost:4317",
        protocol: OTLPProtocol = OTLPProtocol.HTTP,
        headers: dict[str, str] | None = None,
        service_name: str = "liteagent",
    ):
        self.endpoint = endpoint.rstrip("/")
        self.protocol = protocol
        self.headers = headers or {}
        self.service_name = service_name
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.endpoint,
                headers={
                    "Content-Type": "application/json",
                    **self.headers,
                },
                timeout=30.0,
            )
        return self._client

    async def export_counter(
        self,
        name: str,
        value: int,
        attributes: dict[str, Any] | None = None,
        description: str = "",
    ) -> bool:
        """Export a counter metric."""
        import time

        metric = {
            "resourceMetrics": [
                {
                    "resource": {
                        "attributes": [
                            {"key": "service.name", "value": {"stringValue": self.service_name}},
                        ],
                    },
                    "scopeMetrics": [
                        {
                            "scope": {"name": "liteagent"},
                            "metrics": [
                                {
                                    "name": name,
                                    "description": description,
                                    "sum": {
                                        "dataPoints": [
                                            {
                                                "asInt": str(value),
                                                "timeUnixNano": str(int(time.time() * 1e9)),
                                                "attributes": self._to_otlp_attributes(attributes or {}),
                                            }
                                        ],
                                        "aggregationTemporality": 2,  # CUMULATIVE
                                        "isMonotonic": True,
                                    },
                                }
                            ],
                        }
                    ],
                }
            ]
        }

        try:
            client = await self._get_client()
            response = await client.post("/v1/metrics", json=metric)
            response.raise_for_status()
            return True
        except Exception as e:
            print(f"Failed to export metric: {e}")
            return False

    def _to_otlp_attributes(self, attrs: dict[str, Any]) -> list[dict[str, Any]]:
        """Convert attributes to OTLP format."""
        result = []
        for key, value in attrs.items():
            if isinstance(value, str):
                result.append({"key": key, "value": {"stringValue": value}})
            elif isinstance(value, int):
                result.append({"key": key, "value": {"intValue": str(value)}})
            elif isinstance(value, float):
                result.append({"key": key, "value": {"doubleValue": value}})
            elif isinstance(value, bool):
                result.append({"key": key, "value": {"boolValue": value}})
        return result

    async def shutdown(self) -> None:
        """Shutdown the exporter."""
        if self._client:
            await self._client.aclose()
            self._client = None
