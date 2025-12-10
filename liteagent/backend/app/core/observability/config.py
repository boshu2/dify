"""
Observability configuration for LiteAgent.

Supports multiple tracing backends and OpenTelemetry export.
"""

from enum import Enum
from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings


class TracingProvider(str, Enum):
    """Supported tracing providers."""

    NONE = "none"
    LANGFUSE = "langfuse"
    LANGSMITH = "langsmith"
    ARIZE_PHOENIX = "arize_phoenix"
    OPIK = "opik"
    WEAVE = "weave"
    MLFLOW = "mlflow"
    OTEL = "otel"  # OpenTelemetry


class OTLPProtocol(str, Enum):
    """OTLP export protocols."""

    GRPC = "grpc"
    HTTP = "http/protobuf"


class ObservabilityConfig(BaseSettings):
    """Observability settings."""

    # General
    enabled: bool = Field(default=True, description="Enable observability features")
    service_name: str = Field(default="liteagent", description="Service name for traces")
    service_version: str = Field(default="1.0.0", description="Service version")
    environment: str = Field(default="development", description="Deployment environment")

    # Tracing Provider
    tracing_provider: TracingProvider = Field(
        default=TracingProvider.NONE,
        description="Primary tracing provider",
    )

    # OpenTelemetry Configuration
    otel_enabled: bool = Field(default=False, description="Enable OpenTelemetry")
    otel_exporter_otlp_endpoint: str = Field(
        default="http://localhost:4317",
        description="OTLP endpoint URL",
    )
    otel_exporter_otlp_protocol: OTLPProtocol = Field(
        default=OTLPProtocol.GRPC,
        description="OTLP protocol",
    )
    otel_exporter_otlp_headers: str = Field(
        default="",
        description="OTLP headers (key=value,key2=value2)",
    )
    otel_traces_sampler: str = Field(
        default="parentbased_traceidratio",
        description="Trace sampler type",
    )
    otel_traces_sampler_arg: float = Field(
        default=1.0,
        description="Sampler argument (e.g., ratio)",
    )
    otel_metrics_enabled: bool = Field(default=True, description="Enable OTEL metrics")
    otel_logs_enabled: bool = Field(default=False, description="Enable OTEL logs")

    # LangFuse Configuration
    langfuse_enabled: bool = Field(default=False, description="Enable LangFuse tracing")
    langfuse_public_key: str = Field(default="", description="LangFuse public key")
    langfuse_secret_key: str = Field(default="", description="LangFuse secret key")
    langfuse_host: str = Field(
        default="https://cloud.langfuse.com",
        description="LangFuse host URL",
    )

    # LangSmith Configuration
    langsmith_enabled: bool = Field(default=False, description="Enable LangSmith tracing")
    langsmith_api_key: str = Field(default="", description="LangSmith API key")
    langsmith_project: str = Field(default="default", description="LangSmith project name")
    langsmith_endpoint: str = Field(
        default="https://api.smith.langchain.com",
        description="LangSmith API endpoint",
    )

    # Arize Phoenix Configuration
    arize_phoenix_enabled: bool = Field(default=False, description="Enable Arize Phoenix")
    arize_phoenix_endpoint: str = Field(
        default="http://localhost:6006",
        description="Phoenix collector endpoint",
    )

    # Opik Configuration
    opik_enabled: bool = Field(default=False, description="Enable Opik tracing")
    opik_api_key: str = Field(default="", description="Opik API key")
    opik_workspace: str = Field(default="", description="Opik workspace")
    opik_project: str = Field(default="default", description="Opik project name")

    # Weave (W&B) Configuration
    weave_enabled: bool = Field(default=False, description="Enable Weave tracing")
    wandb_api_key: str = Field(default="", description="Weights & Biases API key")
    wandb_project: str = Field(default="liteagent", description="W&B project name")
    wandb_entity: str = Field(default="", description="W&B entity/team name")

    # MLflow Configuration
    mlflow_enabled: bool = Field(default=False, description="Enable MLflow tracing")
    mlflow_tracking_uri: str = Field(
        default="http://localhost:5000",
        description="MLflow tracking server URI",
    )
    mlflow_experiment_name: str = Field(
        default="liteagent",
        description="MLflow experiment name",
    )

    # Sentry Configuration
    sentry_enabled: bool = Field(default=False, description="Enable Sentry error tracking")
    sentry_dsn: str = Field(default="", description="Sentry DSN")
    sentry_traces_sample_rate: float = Field(
        default=0.1,
        description="Sentry traces sample rate (0.0-1.0)",
    )
    sentry_profiles_sample_rate: float = Field(
        default=0.1,
        description="Sentry profiles sample rate (0.0-1.0)",
    )
    sentry_send_default_pii: bool = Field(
        default=False,
        description="Send PII to Sentry",
    )

    # Logging Configuration
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Log level",
    )
    log_format: Literal["json", "text"] = Field(
        default="json",
        description="Log format",
    )
    log_file_enabled: bool = Field(default=False, description="Enable file logging")
    log_file_path: str = Field(default="logs/liteagent.log", description="Log file path")
    log_file_max_bytes: int = Field(
        default=10 * 1024 * 1024,  # 10MB
        description="Max log file size before rotation",
    )
    log_file_backup_count: int = Field(
        default=5,
        description="Number of backup log files to keep",
    )
    log_include_request_body: bool = Field(
        default=False,
        description="Include request body in logs (may contain sensitive data)",
    )

    # Request ID Configuration
    request_id_header: str = Field(
        default="X-Request-ID",
        description="Header name for request ID",
    )
    generate_request_id: bool = Field(
        default=True,
        description="Generate request ID if not provided",
    )

    # Trace Types to Capture
    trace_llm_calls: bool = Field(default=True, description="Trace LLM API calls")
    trace_tool_calls: bool = Field(default=True, description="Trace tool executions")
    trace_retrieval: bool = Field(default=True, description="Trace RAG retrieval")
    trace_workflows: bool = Field(default=True, description="Trace workflow execution")
    trace_agent_steps: bool = Field(default=True, description="Trace agent steps")

    class Config:
        env_prefix = "LITEAGENT_"
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache
def get_observability_config() -> ObservabilityConfig:
    """Get cached observability configuration."""
    return ObservabilityConfig()
