"""
Sandbox module for safe code execution.

Provides isolated execution environment for:
- Python code
- JavaScript/Node.js code
- Jinja2 templates
- Plugin/tool installation

Architecture:
- Production: Use SandboxClient to call standalone sandbox service
- Development: Use CodeExecutor for local execution (RestrictedPython)
"""

from app.core.sandbox.executor import (
    CodeExecutor,
    CodeLanguage,
    ExecutionResult,
    ExecutionError,
    get_executor,
)
from app.core.sandbox.config import SandboxConfig, get_sandbox_config
from app.core.sandbox.transformers import (
    CodeTransformer,
    PythonTransformer,
    JavaScriptTransformer,
    Jinja2Transformer,
)
from app.core.sandbox.validator import (
    OutputValidator,
    ValidationError,
)
from app.core.sandbox.plugins import (
    PluginManager,
    Plugin,
    PluginMetadata,
    PluginStatus,
    PluginType,
)
from app.core.sandbox.client import (
    SandboxClient,
    SandboxResponse,
    get_sandbox_client,
)

__all__ = [
    # Client (for production - calls standalone service)
    "SandboxClient",
    "SandboxResponse",
    "get_sandbox_client",
    # Executor (for local development)
    "CodeExecutor",
    "CodeLanguage",
    "ExecutionResult",
    "ExecutionError",
    "get_executor",
    # Config
    "SandboxConfig",
    "get_sandbox_config",
    # Transformers
    "CodeTransformer",
    "PythonTransformer",
    "JavaScriptTransformer",
    "Jinja2Transformer",
    # Validator
    "OutputValidator",
    "ValidationError",
    # Plugins
    "PluginManager",
    "Plugin",
    "PluginMetadata",
    "PluginStatus",
    "PluginType",
]
