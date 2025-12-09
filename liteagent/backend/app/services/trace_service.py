"""
Trace management service for LiteAgent.

Provides management of tracing configurations and trace data.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from sqlalchemy import Column, DateTime, Enum as SQLEnum, String, Text, Boolean
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import Base


class TraceProviderType(str, Enum):
    """Supported trace provider types."""

    LANGFUSE = "langfuse"
    LANGSMITH = "langsmith"
    ARIZE_PHOENIX = "arize_phoenix"
    OPIK = "opik"
    WEAVE = "weave"
    OTEL = "otel"


class TraceConfig(Base):
    """Database model for trace configuration."""

    __tablename__ = "trace_configs"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    tenant_id = Column(String(36), nullable=True, index=True)
    app_id = Column(String(36), nullable=True, index=True)
    provider_type = Column(SQLEnum(TraceProviderType), nullable=False)
    enabled = Column(Boolean, default=True)

    # Provider-specific configuration (JSON stored as text)
    config = Column(Text, nullable=False, default="{}")

    # Trace filters
    trace_llm = Column(Boolean, default=True)
    trace_tools = Column(Boolean, default=True)
    trace_retrieval = Column(Boolean, default=True)
    trace_workflows = Column(Boolean, default=True)
    trace_agents = Column(Boolean, default=True)

    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        import json

        return {
            "id": self.id,
            "tenant_id": self.tenant_id,
            "app_id": self.app_id,
            "provider_type": self.provider_type.value,
            "enabled": self.enabled,
            "config": json.loads(self.config) if self.config else {},
            "trace_llm": self.trace_llm,
            "trace_tools": self.trace_tools,
            "trace_retrieval": self.trace_retrieval,
            "trace_workflows": self.trace_workflows,
            "trace_agents": self.trace_agents,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


@dataclass
class TraceConfigCreate:
    """Input for creating a trace configuration."""

    provider_type: TraceProviderType
    config: dict[str, Any] = field(default_factory=dict)
    tenant_id: str | None = None
    app_id: str | None = None
    enabled: bool = True
    trace_llm: bool = True
    trace_tools: bool = True
    trace_retrieval: bool = True
    trace_workflows: bool = True
    trace_agents: bool = True


@dataclass
class TraceConfigUpdate:
    """Input for updating a trace configuration."""

    config: dict[str, Any] | None = None
    enabled: bool | None = None
    trace_llm: bool | None = None
    trace_tools: bool | None = None
    trace_retrieval: bool | None = None
    trace_workflows: bool | None = None
    trace_agents: bool | None = None


class TraceService:
    """Service for managing trace configurations."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create_config(self, data: TraceConfigCreate) -> TraceConfig:
        """Create a new trace configuration."""
        import json

        from sqlalchemy import select

        # Check for existing config
        query = select(TraceConfig).where(
            TraceConfig.provider_type == data.provider_type,
        )
        if data.tenant_id:
            query = query.where(TraceConfig.tenant_id == data.tenant_id)
        if data.app_id:
            query = query.where(TraceConfig.app_id == data.app_id)

        result = await self.session.execute(query)
        existing = result.scalar_one_or_none()

        if existing:
            raise ValueError(f"Trace config for {data.provider_type.value} already exists")

        config = TraceConfig(
            provider_type=data.provider_type,
            tenant_id=data.tenant_id,
            app_id=data.app_id,
            enabled=data.enabled,
            config=json.dumps(data.config),
            trace_llm=data.trace_llm,
            trace_tools=data.trace_tools,
            trace_retrieval=data.trace_retrieval,
            trace_workflows=data.trace_workflows,
            trace_agents=data.trace_agents,
        )

        self.session.add(config)
        await self.session.commit()
        await self.session.refresh(config)

        return config

    async def get_config(self, config_id: str) -> TraceConfig | None:
        """Get a trace configuration by ID."""
        from sqlalchemy import select

        query = select(TraceConfig).where(TraceConfig.id == config_id)
        result = await self.session.execute(query)
        return result.scalar_one_or_none()

    async def get_config_by_provider(
        self,
        provider_type: TraceProviderType,
        tenant_id: str | None = None,
        app_id: str | None = None,
    ) -> TraceConfig | None:
        """Get trace configuration by provider type."""
        from sqlalchemy import select

        query = select(TraceConfig).where(
            TraceConfig.provider_type == provider_type,
        )
        if tenant_id:
            query = query.where(TraceConfig.tenant_id == tenant_id)
        if app_id:
            query = query.where(TraceConfig.app_id == app_id)

        result = await self.session.execute(query)
        return result.scalar_one_or_none()

    async def list_configs(
        self,
        tenant_id: str | None = None,
        app_id: str | None = None,
        enabled_only: bool = False,
    ) -> list[TraceConfig]:
        """List trace configurations."""
        from sqlalchemy import select

        query = select(TraceConfig)

        if tenant_id:
            query = query.where(TraceConfig.tenant_id == tenant_id)
        if app_id:
            query = query.where(TraceConfig.app_id == app_id)
        if enabled_only:
            query = query.where(TraceConfig.enabled == True)  # noqa: E712

        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def update_config(
        self,
        config_id: str,
        data: TraceConfigUpdate,
    ) -> TraceConfig | None:
        """Update a trace configuration."""
        import json

        config = await self.get_config(config_id)
        if not config:
            return None

        if data.config is not None:
            config.config = json.dumps(data.config)
        if data.enabled is not None:
            config.enabled = data.enabled
        if data.trace_llm is not None:
            config.trace_llm = data.trace_llm
        if data.trace_tools is not None:
            config.trace_tools = data.trace_tools
        if data.trace_retrieval is not None:
            config.trace_retrieval = data.trace_retrieval
        if data.trace_workflows is not None:
            config.trace_workflows = data.trace_workflows
        if data.trace_agents is not None:
            config.trace_agents = data.trace_agents

        await self.session.commit()
        await self.session.refresh(config)

        return config

    async def delete_config(self, config_id: str) -> bool:
        """Delete a trace configuration."""
        config = await self.get_config(config_id)
        if not config:
            return False

        await self.session.delete(config)
        await self.session.commit()
        return True

    async def toggle_config(self, config_id: str, enabled: bool) -> TraceConfig | None:
        """Enable or disable a trace configuration."""
        config = await self.get_config(config_id)
        if not config:
            return None

        config.enabled = enabled
        await self.session.commit()
        await self.session.refresh(config)

        return config


class TraceManager:
    """Manager for trace exporters based on configuration."""

    def __init__(self):
        self._exporters: dict[str, Any] = {}

    def get_exporter(self, config: TraceConfig) -> Any:
        """Get or create an exporter for the given configuration."""
        import json

        config_data = json.loads(config.config) if config.config else {}

        if config.id in self._exporters:
            return self._exporters[config.id]

        exporter = self._create_exporter(config.provider_type, config_data)
        if exporter:
            self._exporters[config.id] = exporter

        return exporter

    def _create_exporter(
        self,
        provider_type: TraceProviderType,
        config: dict[str, Any],
    ) -> Any:
        """Create an exporter for the given provider type."""
        if provider_type == TraceProviderType.LANGFUSE:
            from app.core.observability.exporters.langfuse import LangFuseExporter

            return LangFuseExporter(
                public_key=config.get("public_key", ""),
                secret_key=config.get("secret_key", ""),
                host=config.get("host", "https://cloud.langfuse.com"),
            )

        elif provider_type == TraceProviderType.LANGSMITH:
            from app.core.observability.exporters.langsmith import LangSmithExporter

            return LangSmithExporter(
                api_key=config.get("api_key", ""),
                project_name=config.get("project", "default"),
                endpoint=config.get("endpoint", "https://api.smith.langchain.com"),
            )

        elif provider_type == TraceProviderType.ARIZE_PHOENIX:
            from app.core.observability.exporters.arize_phoenix import ArizePhoenixExporter

            return ArizePhoenixExporter(
                endpoint=config.get("endpoint", "http://localhost:6006"),
                project_name=config.get("project", "default"),
            )

        elif provider_type == TraceProviderType.OPIK:
            from app.core.observability.exporters.opik import OpikExporter

            return OpikExporter(
                api_key=config.get("api_key", ""),
                workspace=config.get("workspace", ""),
                project_name=config.get("project", "default"),
            )

        elif provider_type == TraceProviderType.WEAVE:
            from app.core.observability.exporters.weave import WeaveExporter

            return WeaveExporter(
                api_key=config.get("api_key", ""),
                project_name=config.get("project", "liteagent"),
                entity=config.get("entity", ""),
            )

        elif provider_type == TraceProviderType.OTEL:
            from app.core.observability.exporters.otel import OTLPExporter
            from app.core.observability.config import OTLPProtocol

            return OTLPExporter(
                endpoint=config.get("endpoint", "http://localhost:4317"),
                protocol=OTLPProtocol(config.get("protocol", "grpc")),
                headers=config.get("headers", {}),
                service_name=config.get("service_name", "liteagent"),
            )

        return None

    async def shutdown(self) -> None:
        """Shutdown all exporters."""
        for exporter in self._exporters.values():
            if hasattr(exporter, "shutdown"):
                await exporter.shutdown()
        self._exporters.clear()


# Global trace manager
_trace_manager: TraceManager | None = None


def get_trace_manager() -> TraceManager:
    """Get the global trace manager."""
    global _trace_manager
    if _trace_manager is None:
        _trace_manager = TraceManager()
    return _trace_manager
