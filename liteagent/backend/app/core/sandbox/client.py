"""
Sandbox service client.

HTTP client for communicating with the standalone sandbox service.
"""

import json
import re
from dataclasses import dataclass
from typing import Any

import httpx

from app.core.sandbox.config import SandboxConfig, get_sandbox_config


@dataclass
class SandboxResponse:
    """Response from sandbox service."""

    success: bool
    output: Any = None
    stdout: str = ""
    stderr: str = ""
    error: str | None = None
    error_type: str | None = None
    execution_time_ms: float = 0


class SandboxClient:
    """
    HTTP client for sandbox service.

    Provides a clean interface to the standalone sandbox microservice.
    """

    RESULT_START = "<<RESULT>>"
    RESULT_END = "<<RESULT>>"

    def __init__(self, config: SandboxConfig | None = None):
        self.config = config or get_sandbox_config()
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.config.remote_endpoint,
                timeout=httpx.Timeout(
                    connect=self.config.connection_timeout,
                    read=self.config.execution_timeout + 10,
                    write=10.0,
                    pool=5.0,
                ),
                limits=httpx.Limits(
                    max_connections=100,
                    max_keepalive_connections=20,
                    keepalive_expiry=5.0,
                ),
            )
        return self._client

    async def execute(
        self,
        language: str,
        code: str,
        inputs: dict[str, Any] | None = None,
        preload: str | None = None,
        enable_network: bool = False,
    ) -> SandboxResponse:
        """
        Execute code via sandbox service.

        Args:
            language: Programming language (python3, javascript, jinja2)
            code: Code to execute
            inputs: Input variables (encoded in preload for compatibility)
            preload: Preload script
            enable_network: Enable network access

        Returns:
            SandboxResponse with execution results
        """
        client = await self._get_client()

        # Encode inputs in preload for backwards compatibility
        actual_preload = preload or ""
        if inputs:
            import base64

            inputs_b64 = base64.b64encode(json.dumps(inputs).encode()).decode()
            # The transformer in sandbox service will decode this
            actual_preload = f"# INPUTS_B64: {inputs_b64}\n{actual_preload}"

        try:
            response = await client.post(
                "/v1/sandbox/run",
                json={
                    "language": language,
                    "code": code,
                    "preload": actual_preload,
                    "enable_network": enable_network,
                },
                headers={"X-Api-Key": self.config.remote_api_key},
            )

            if response.status_code == 401:
                return SandboxResponse(
                    success=False,
                    error="Invalid sandbox API key",
                    error_type="AuthenticationError",
                )

            if response.status_code == 503:
                return SandboxResponse(
                    success=False,
                    error="Sandbox service unavailable",
                    error_type="ServiceUnavailableError",
                )

            data = response.json()

            if data.get("code") == 0:
                # Success - extract result from stdout
                stdout = data.get("data", {}).get("stdout", "")
                output = self._extract_result(stdout)

                return SandboxResponse(
                    success=True,
                    output=output,
                    stdout=stdout,
                    stderr=data.get("data", {}).get("error", ""),
                )
            else:
                return SandboxResponse(
                    success=False,
                    error=data.get("message", "Execution failed"),
                    error_type="ExecutionError",
                    stdout=data.get("data", {}).get("stdout", ""),
                    stderr=data.get("data", {}).get("error", ""),
                )

        except httpx.ConnectError:
            return SandboxResponse(
                success=False,
                error="Cannot connect to sandbox service",
                error_type="ConnectionError",
            )
        except httpx.TimeoutException:
            return SandboxResponse(
                success=False,
                error="Sandbox execution timed out",
                error_type="TimeoutError",
            )
        except Exception as e:
            return SandboxResponse(
                success=False,
                error=str(e),
                error_type=type(e).__name__,
            )

    def _extract_result(self, output: str) -> Any:
        """Extract result from output markers."""
        pattern = f"{self.RESULT_START}(.+?){self.RESULT_END}"
        match = re.search(pattern, output, re.DOTALL)

        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                return match.group(1)

        return None

    async def health_check(self) -> bool:
        """Check if sandbox service is healthy."""
        try:
            client = await self._get_client()
            response = await client.get("/health")
            return response.status_code == 200
        except Exception:
            return False

    async def list_languages(self) -> list[dict[str, Any]]:
        """Get list of supported languages."""
        try:
            client = await self._get_client()
            response = await client.get(
                "/v1/sandbox/languages",
                headers={"X-Api-Key": self.config.remote_api_key},
            )
            data = response.json()
            return data.get("languages", [])
        except Exception:
            return []

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None


# Global client instance
_client: SandboxClient | None = None


def get_sandbox_client() -> SandboxClient:
    """Get global sandbox client."""
    global _client
    if _client is None:
        _client = SandboxClient()
    return _client
