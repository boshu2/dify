"""
Sandbox configuration for code execution.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings


class SandboxConfig(BaseSettings):
    """Configuration for sandbox code execution."""

    # Execution mode
    mode: Literal["local", "docker", "remote"] = Field(
        default="local",
        description="Execution mode: local (RestrictedPython), docker (container), remote (HTTP API)",
    )

    # Remote sandbox settings (when mode=remote)
    remote_endpoint: str = Field(
        default="http://localhost:8194",
        description="Remote sandbox API endpoint",
    )
    remote_api_key: str = Field(
        default="liteagent-sandbox",
        description="API key for remote sandbox",
    )

    # Docker sandbox settings (when mode=docker)
    docker_image: str = Field(
        default="liteagent/sandbox:latest",
        description="Docker image for sandbox",
    )
    docker_network: str = Field(
        default="sandbox_network",
        description="Docker network for sandbox isolation",
    )

    # Timeout settings
    execution_timeout: int = Field(
        default=30,
        description="Code execution timeout in seconds",
    )
    connection_timeout: int = Field(
        default=10,
        description="Connection timeout for remote sandbox",
    )

    # Resource limits
    max_workers: int = Field(
        default=4,
        description="Maximum concurrent code executions",
    )
    max_memory_mb: int = Field(
        default=256,
        description="Maximum memory per execution in MB",
    )
    max_cpu_time: int = Field(
        default=30,
        description="Maximum CPU time in seconds",
    )

    # Output limits
    max_output_size: int = Field(
        default=1_000_000,
        description="Maximum output size in bytes (1MB)",
    )
    max_string_length: int = Field(
        default=400_000,
        description="Maximum string length in output",
    )
    max_number: int = Field(
        default=9223372036854775807,
        description="Maximum number value",
    )
    min_number: int = Field(
        default=-9223372036854775807,
        description="Minimum number value",
    )
    max_precision: int = Field(
        default=20,
        description="Maximum decimal precision",
    )
    max_depth: int = Field(
        default=5,
        description="Maximum object nesting depth",
    )
    max_array_length: int = Field(
        default=1000,
        description="Maximum array length",
    )

    # Network settings
    enable_network: bool = Field(
        default=False,
        description="Allow network access in sandbox",
    )
    allowed_hosts: list[str] = Field(
        default=[],
        description="Allowed hosts when network is enabled",
    )
    http_proxy: str = Field(
        default="",
        description="HTTP proxy for sandbox network access",
    )
    https_proxy: str = Field(
        default="",
        description="HTTPS proxy for sandbox network access",
    )

    # Allowed modules/packages (for local mode)
    allowed_modules: list[str] = Field(
        default=[
            "json",
            "math",
            "random",
            "datetime",
            "re",
            "collections",
            "itertools",
            "functools",
            "operator",
            "string",
            "base64",
            "hashlib",
            "urllib.parse",
        ],
        description="Allowed Python modules in local sandbox",
    )

    # Blocked patterns
    blocked_patterns: list[str] = Field(
        default=[
            "import os",
            "import sys",
            "import subprocess",
            "import socket",
            "__import__",
            "eval(",
            "exec(",
            "compile(",
            "open(",
            "file(",
            "input(",
            "breakpoint(",
        ],
        description="Blocked code patterns",
    )

    # Plugin settings
    plugins_dir: str = Field(
        default="plugins",
        description="Directory for installed plugins",
    )
    plugin_registry_url: str = Field(
        default="",
        description="Plugin registry URL for discovery",
    )

    class Config:
        env_prefix = "SANDBOX_"
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache
def get_sandbox_config() -> SandboxConfig:
    """Get cached sandbox configuration."""
    return SandboxConfig()
