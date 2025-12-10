"""
Sandbox module for safe code execution.

Provides isolated execution environment for:
- Python code
- JavaScript/Node.js code
- Jinja2 templates
- Plugin/tool installation
"""

from app.core.sandbox.executor import (
    CodeExecutor,
    CodeLanguage,
    ExecutionResult,
    ExecutionError,
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

__all__ = [
    # Executor
    "CodeExecutor",
    "CodeLanguage",
    "ExecutionResult",
    "ExecutionError",
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
