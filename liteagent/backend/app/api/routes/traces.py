"""
Trace management API routes.

Provides endpoints for managing trace configurations.
"""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.services.trace_service import (
    TraceConfigCreate,
    TraceConfigUpdate,
    TraceProviderType,
    TraceService,
)

router = APIRouter(prefix="/traces", tags=["traces"])


class TraceConfigCreateRequest(BaseModel):
    """Request model for creating a trace configuration."""

    provider_type: str
    config: dict[str, Any] = {}
    tenant_id: str | None = None
    app_id: str | None = None
    enabled: bool = True
    trace_llm: bool = True
    trace_tools: bool = True
    trace_retrieval: bool = True
    trace_workflows: bool = True
    trace_agents: bool = True


class TraceConfigUpdateRequest(BaseModel):
    """Request model for updating a trace configuration."""

    config: dict[str, Any] | None = None
    enabled: bool | None = None
    trace_llm: bool | None = None
    trace_tools: bool | None = None
    trace_retrieval: bool | None = None
    trace_workflows: bool | None = None
    trace_agents: bool | None = None


@router.post("/configs")
async def create_trace_config(
    request: TraceConfigCreateRequest,
    session: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Create a new trace configuration."""
    try:
        provider_type = TraceProviderType(request.provider_type)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid provider type: {request.provider_type}",
        )

    service = TraceService(session)

    try:
        config = await service.create_config(
            TraceConfigCreate(
                provider_type=provider_type,
                config=request.config,
                tenant_id=request.tenant_id,
                app_id=request.app_id,
                enabled=request.enabled,
                trace_llm=request.trace_llm,
                trace_tools=request.trace_tools,
                trace_retrieval=request.trace_retrieval,
                trace_workflows=request.trace_workflows,
                trace_agents=request.trace_agents,
            )
        )
        return config.to_dict()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/configs")
async def list_trace_configs(
    tenant_id: str | None = None,
    app_id: str | None = None,
    enabled_only: bool = False,
    session: AsyncSession = Depends(get_db),
) -> list[dict[str, Any]]:
    """List trace configurations."""
    service = TraceService(session)
    configs = await service.list_configs(
        tenant_id=tenant_id,
        app_id=app_id,
        enabled_only=enabled_only,
    )
    return [c.to_dict() for c in configs]


@router.get("/configs/{config_id}")
async def get_trace_config(
    config_id: str,
    session: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Get a trace configuration by ID."""
    service = TraceService(session)
    config = await service.get_config(config_id)

    if not config:
        raise HTTPException(status_code=404, detail="Trace config not found")

    return config.to_dict()


@router.patch("/configs/{config_id}")
async def update_trace_config(
    config_id: str,
    request: TraceConfigUpdateRequest,
    session: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Update a trace configuration."""
    service = TraceService(session)

    config = await service.update_config(
        config_id,
        TraceConfigUpdate(
            config=request.config,
            enabled=request.enabled,
            trace_llm=request.trace_llm,
            trace_tools=request.trace_tools,
            trace_retrieval=request.trace_retrieval,
            trace_workflows=request.trace_workflows,
            trace_agents=request.trace_agents,
        ),
    )

    if not config:
        raise HTTPException(status_code=404, detail="Trace config not found")

    return config.to_dict()


@router.delete("/configs/{config_id}")
async def delete_trace_config(
    config_id: str,
    session: AsyncSession = Depends(get_db),
) -> dict[str, str]:
    """Delete a trace configuration."""
    service = TraceService(session)
    deleted = await service.delete_config(config_id)

    if not deleted:
        raise HTTPException(status_code=404, detail="Trace config not found")

    return {"status": "deleted"}


@router.post("/configs/{config_id}/enable")
async def enable_trace_config(
    config_id: str,
    session: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Enable a trace configuration."""
    service = TraceService(session)
    config = await service.toggle_config(config_id, enabled=True)

    if not config:
        raise HTTPException(status_code=404, detail="Trace config not found")

    return config.to_dict()


@router.post("/configs/{config_id}/disable")
async def disable_trace_config(
    config_id: str,
    session: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Disable a trace configuration."""
    service = TraceService(session)
    config = await service.toggle_config(config_id, enabled=False)

    if not config:
        raise HTTPException(status_code=404, detail="Trace config not found")

    return config.to_dict()


@router.get("/providers")
async def list_trace_providers() -> list[dict[str, Any]]:
    """List available trace providers."""
    providers = [
        {
            "type": TraceProviderType.LANGFUSE.value,
            "name": "LangFuse",
            "description": "LLM observability with prompt management and evaluation",
            "config_schema": {
                "public_key": {"type": "string", "required": True},
                "secret_key": {"type": "string", "required": True, "secret": True},
                "host": {"type": "string", "required": False, "default": "https://cloud.langfuse.com"},
            },
        },
        {
            "type": TraceProviderType.LANGSMITH.value,
            "name": "LangSmith",
            "description": "LangChain ecosystem tracing and evaluation",
            "config_schema": {
                "api_key": {"type": "string", "required": True, "secret": True},
                "project": {"type": "string", "required": False, "default": "default"},
                "endpoint": {"type": "string", "required": False, "default": "https://api.smith.langchain.com"},
            },
        },
        {
            "type": TraceProviderType.ARIZE_PHOENIX.value,
            "name": "Arize Phoenix",
            "description": "Local-first ML observability and evaluation",
            "config_schema": {
                "endpoint": {"type": "string", "required": False, "default": "http://localhost:6006"},
                "project": {"type": "string", "required": False, "default": "default"},
            },
        },
        {
            "type": TraceProviderType.OPIK.value,
            "name": "Opik (Comet ML)",
            "description": "Comet ML LLM observability and evaluation",
            "config_schema": {
                "api_key": {"type": "string", "required": True, "secret": True},
                "workspace": {"type": "string", "required": False},
                "project": {"type": "string", "required": False, "default": "default"},
            },
        },
        {
            "type": TraceProviderType.WEAVE.value,
            "name": "Weave (W&B)",
            "description": "Weights & Biases LLM tracing and evaluation",
            "config_schema": {
                "api_key": {"type": "string", "required": True, "secret": True},
                "project": {"type": "string", "required": False, "default": "liteagent"},
                "entity": {"type": "string", "required": False},
            },
        },
        {
            "type": TraceProviderType.OTEL.value,
            "name": "OpenTelemetry",
            "description": "Standard observability with OTLP export",
            "config_schema": {
                "endpoint": {"type": "string", "required": False, "default": "http://localhost:4317"},
                "protocol": {"type": "string", "required": False, "default": "grpc", "enum": ["grpc", "http/protobuf"]},
                "headers": {"type": "object", "required": False},
                "service_name": {"type": "string", "required": False, "default": "liteagent"},
            },
        },
    ]
    return providers
