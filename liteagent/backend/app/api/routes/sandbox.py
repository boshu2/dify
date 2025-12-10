"""
Sandbox API endpoints for code execution and plugin management.
"""

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.core.sandbox import (
    CodeExecutor,
    CodeLanguage,
    OutputValidator,
    PluginManager,
    PluginStatus,
    PluginType,
)
from app.core.sandbox.executor import get_executor
from app.core.sandbox.plugins import get_plugin_manager

router = APIRouter(prefix="/sandbox", tags=["sandbox"])


# Request/Response Models
class ExecuteCodeRequest(BaseModel):
    """Request model for code execution."""

    language: str = Field(
        description="Programming language (python3, javascript, jinja2)"
    )
    code: str = Field(description="Code to execute")
    inputs: dict[str, Any] = Field(default={}, description="Input variables")
    preload: str | None = Field(default=None, description="Preload script")
    enable_network: bool | None = Field(default=None, description="Allow network access")
    timeout: int | None = Field(default=None, description="Execution timeout in seconds")


class ExecuteCodeResponse(BaseModel):
    """Response model for code execution."""

    success: bool
    output: Any = None
    stdout: str = ""
    stderr: str = ""
    error: str | None = None
    error_type: str | None = None
    execution_time_ms: float = 0


class PluginInstallRequest(BaseModel):
    """Request model for plugin installation."""

    source: str = Field(description="Source type: pypi, git, or local")
    package: str = Field(description="Package name, git URL, or local path")
    version: str | None = Field(default=None, description="Version (for PyPI)")
    ref: str = Field(default="main", description="Git ref (for git source)")


class PluginConfigRequest(BaseModel):
    """Request model for plugin configuration."""

    config: dict[str, Any] = Field(description="Plugin configuration")


# Code Execution Endpoints
@router.post("/execute", response_model=ExecuteCodeResponse)
async def execute_code(request: ExecuteCodeRequest) -> ExecuteCodeResponse:
    """
    Execute code in the sandbox.

    Supports Python 3, JavaScript, and Jinja2 templates.
    """
    try:
        language = CodeLanguage(request.language)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid language: {request.language}. "
            f"Supported: {[l.value for l in CodeLanguage]}",
        )

    executor = get_executor()

    result = await executor.execute(
        language=language,
        code=request.code,
        inputs=request.inputs,
        preload=request.preload,
        enable_network=request.enable_network,
        timeout=request.timeout,
    )

    # Validate output
    if result.success and result.output is not None:
        validator = OutputValidator()
        validation_result = validator.validate(result.output)

        if not validation_result.valid:
            result.success = False
            result.error = validation_result.error
            result.error_type = "ValidationError"
        else:
            result.output = validation_result.value

    return ExecuteCodeResponse(
        success=result.success,
        output=result.output,
        stdout=result.stdout,
        stderr=result.stderr,
        error=result.error,
        error_type=result.error_type,
        execution_time_ms=result.execution_time_ms,
    )


@router.get("/languages")
async def list_languages() -> list[dict[str, str]]:
    """List supported programming languages."""
    return [
        {
            "id": CodeLanguage.PYTHON3.value,
            "name": "Python 3",
            "description": "Python 3.11+ with restricted builtins",
            "file_extension": ".py",
        },
        {
            "id": CodeLanguage.JAVASCRIPT.value,
            "name": "JavaScript",
            "description": "Node.js runtime",
            "file_extension": ".js",
        },
        {
            "id": CodeLanguage.JINJA2.value,
            "name": "Jinja2",
            "description": "Jinja2 template engine (sandboxed)",
            "file_extension": ".j2",
        },
    ]


# Plugin Management Endpoints
@router.get("/plugins")
async def list_plugins(
    status: str | None = None,
    plugin_type: str | None = None,
) -> list[dict[str, Any]]:
    """List installed plugins."""
    manager = get_plugin_manager()

    status_filter = PluginStatus(status) if status else None
    type_filter = PluginType(plugin_type) if plugin_type else None

    plugins = manager.list_plugins(status=status_filter, plugin_type=type_filter)

    return [p.to_dict() for p in plugins]


@router.post("/plugins/install")
async def install_plugin(request: PluginInstallRequest) -> dict[str, Any]:
    """
    Install a plugin.

    Sources:
    - pypi: Install from PyPI (package = package name)
    - git: Install from git repository (package = git URL)
    - local: Install from local directory (package = directory path)
    """
    manager = get_plugin_manager()

    try:
        if request.source == "pypi":
            plugin = await manager.install_from_pypi(
                package_name=request.package,
                version=request.version,
            )
        elif request.source == "git":
            plugin = await manager.install_from_git(
                repo_url=request.package,
                ref=request.ref,
            )
        elif request.source == "local":
            plugin = await manager.install_from_directory(request.package)
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid source: {request.source}. Use: pypi, git, or local",
            )

        return plugin.to_dict()

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/plugins/{plugin_id}")
async def get_plugin(plugin_id: str) -> dict[str, Any]:
    """Get plugin details."""
    manager = get_plugin_manager()
    plugin = manager.get_plugin(plugin_id)

    if not plugin:
        raise HTTPException(status_code=404, detail="Plugin not found")

    return plugin.to_dict()


@router.delete("/plugins/{plugin_id}")
async def uninstall_plugin(plugin_id: str) -> dict[str, str]:
    """Uninstall a plugin."""
    manager = get_plugin_manager()

    if not manager.uninstall(plugin_id):
        raise HTTPException(status_code=404, detail="Plugin not found")

    return {"status": "uninstalled"}


@router.post("/plugins/{plugin_id}/enable")
async def enable_plugin(plugin_id: str) -> dict[str, Any]:
    """Enable a plugin."""
    manager = get_plugin_manager()
    plugin = manager.enable(plugin_id)

    if not plugin:
        raise HTTPException(status_code=404, detail="Plugin not found")

    if plugin.status == PluginStatus.ERROR:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to enable plugin: {plugin.error_message}",
        )

    return plugin.to_dict()


@router.post("/plugins/{plugin_id}/disable")
async def disable_plugin(plugin_id: str) -> dict[str, Any]:
    """Disable a plugin."""
    manager = get_plugin_manager()
    plugin = manager.disable(plugin_id)

    if not plugin:
        raise HTTPException(status_code=404, detail="Plugin not found")

    return plugin.to_dict()


@router.patch("/plugins/{plugin_id}/config")
async def configure_plugin(
    plugin_id: str,
    request: PluginConfigRequest,
) -> dict[str, Any]:
    """Configure a plugin."""
    manager = get_plugin_manager()
    plugin = manager.configure_plugin(plugin_id, request.config)

    if not plugin:
        raise HTTPException(status_code=404, detail="Plugin not found")

    return plugin.to_dict()


@router.get("/plugins/search")
async def search_plugins(
    query: str = "",
    plugin_type: str | None = None,
    tags: str | None = None,
) -> list[dict[str, Any]]:
    """Search for plugins in registry."""
    manager = get_plugin_manager()

    type_filter = PluginType(plugin_type) if plugin_type else None
    tag_list = tags.split(",") if tags else None

    results = await manager.search_plugins(
        query=query,
        plugin_type=type_filter,
        tags=tag_list,
    )

    return [r.to_dict() for r in results]


@router.get("/plugins/{plugin_id}/tools")
async def get_plugin_tools(plugin_id: str) -> list[dict[str, Any]]:
    """Get tools provided by a plugin."""
    manager = get_plugin_manager()
    plugin = manager.get_plugin(plugin_id)

    if not plugin:
        raise HTTPException(status_code=404, detail="Plugin not found")

    if plugin.status != PluginStatus.ENABLED:
        raise HTTPException(status_code=400, detail="Plugin not enabled")

    if not plugin.module or not hasattr(plugin.module, "get_tools"):
        return []

    tools = plugin.module.get_tools()
    return [
        {
            "name": t.name,
            "description": t.description,
            "parameters": t.parameters if hasattr(t, "parameters") else {},
        }
        for t in tools
    ]


# Health & Status Endpoints
@router.get("/health")
async def sandbox_health() -> dict[str, Any]:
    """Check sandbox health."""
    from app.core.sandbox.config import get_sandbox_config

    config = get_sandbox_config()

    return {
        "status": "healthy",
        "mode": config.mode,
        "languages": [l.value for l in CodeLanguage],
        "max_workers": config.max_workers,
        "execution_timeout": config.execution_timeout,
        "network_enabled": config.enable_network,
    }


@router.get("/config")
async def get_sandbox_config_info() -> dict[str, Any]:
    """Get sandbox configuration (non-sensitive)."""
    from app.core.sandbox.config import get_sandbox_config

    config = get_sandbox_config()

    return {
        "mode": config.mode,
        "execution_timeout": config.execution_timeout,
        "max_workers": config.max_workers,
        "max_memory_mb": config.max_memory_mb,
        "max_output_size": config.max_output_size,
        "enable_network": config.enable_network,
        "allowed_modules": config.allowed_modules,
        "plugins_dir": config.plugins_dir,
    }
