"""
Sandbox service configuration.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings


class SandboxConfig(BaseSettings):
    """Configuration for the sandbox service."""

    model_config = {"env_prefix": "SANDBOX_"}

    # Server settings
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8194)
    debug: bool = Field(default=False)
    api_key: str = Field(default="dify-sandbox")

    # Worker settings
    max_workers: int = Field(default=4)
    max_requests_per_worker: int = Field(default=50)
    worker_timeout: int = Field(default=30)

    # Execution settings
    execution_timeout: int = Field(default=30)
    max_memory_mb: int = Field(default=256)
    max_output_size: int = Field(default=1_000_000)

    # Network settings
    enable_network: bool = Field(default=False)
    allowed_hosts: list[str] = Field(default_factory=list)
    http_proxy: str = Field(default="")
    https_proxy: str = Field(default="")
    socks5_proxy: str = Field(default="")

    # Security settings
    allowed_syscalls: list[str] = Field(default_factory=list)
    blocked_patterns: list[str] = Field(
        default=[
            "import os",
            "import sys",
            "import subprocess",
            "import socket",
            "import shutil",
            "import tempfile",
            "__import__",
            "eval(",
            "exec(",
            "compile(",
            "open(",
            "file(",
            "input(",
            "breakpoint(",
            "globals(",
            "locals(",
            "vars(",
            "dir(",
            "getattr(",
            "setattr(",
            "delattr(",
            "hasattr(",
        ]
    )

    # Allowed modules for RestrictedPython
    allowed_modules: list[str] = Field(
        default=[
            "json",
            "math",
            "random",
            "datetime",
            "time",
            "re",
            "collections",
            "itertools",
            "functools",
            "operator",
            "string",
            "base64",
            "hashlib",
            "hmac",
            "urllib.parse",
            "decimal",
            "fractions",
            "statistics",
            "copy",
            "typing",
        ]
    )

    # Output validation limits
    max_string_length: int = Field(default=400_000)
    max_number: int = Field(default=9_223_372_036_854_775_807)
    min_number: int = Field(default=-9_223_372_036_854_775_807)
    max_precision: int = Field(default=20)
    max_depth: int = Field(default=5)
    max_array_length: int = Field(default=1000)
    max_string_array_length: int = Field(default=30)
    max_number_array_length: int = Field(default=1000)
    max_object_array_length: int = Field(default=30)

    # Runtime paths
    python_path: str = Field(default="/usr/local/bin/python3")
    node_path: str = Field(default="/usr/local/bin/node")


@lru_cache
def get_config() -> SandboxConfig:
    """Get cached configuration."""
    return SandboxConfig()
