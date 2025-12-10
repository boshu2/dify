"""
Structured logging and metrics collection for LiteAgent.
Provides JSON-formatted logs for production and text logs for development.

Features:
- JSON and text log formats
- File rotation with configurable size and backup count
- Request ID propagation
- Context-aware logging
- Specialized loggers for HTTP, LLM, workflow, and retrieval operations
"""

import json
import logging
import os
import sys
import threading
import traceback
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from pathlib import Path
from typing import IO, Any
from uuid import uuid4

# Context variable for log context
_log_context: ContextVar[dict[str, Any]] = ContextVar("log_context", default={})

# Context variable for request ID
_request_id: ContextVar[str | None] = ContextVar("request_id", default=None)

# Context variable for trace ID
_trace_id: ContextVar[str | None] = ContextVar("trace_id", default=None)


@dataclass
class LogConfig:
    """Configuration for logging."""

    level: str = "INFO"
    format: str = "json"  # "json" or "text"
    include_timestamp: bool = True

    # File logging
    file_enabled: bool = False
    file_path: str = "logs/liteagent.log"
    file_max_bytes: int = 10 * 1024 * 1024  # 10MB
    file_backup_count: int = 5
    file_rotation_type: str = "size"  # "size" or "time"
    file_rotation_when: str = "midnight"  # For time-based rotation
    file_rotation_interval: int = 1

    # Request logging
    include_request_body: bool = False
    include_response_body: bool = False
    max_body_length: int = 1000  # Truncate bodies longer than this

    # Timezone
    timezone: str = "UTC"

    def __post_init__(self):
        """Validate configuration."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if self.level.upper() not in valid_levels:
            raise ValueError(f"Invalid log level: {self.level}. Must be one of {valid_levels}")
        self.level = self.level.upper()

    @classmethod
    def from_env(cls) -> "LogConfig":
        """Create configuration from environment variables."""
        return cls(
            level=os.environ.get("LOG_LEVEL", "INFO"),
            format=os.environ.get("LOG_FORMAT", "json"),
            include_timestamp=os.environ.get("LOG_INCLUDE_TIMESTAMP", "true").lower() == "true",
            file_enabled=os.environ.get("LOG_FILE_ENABLED", "false").lower() == "true",
            file_path=os.environ.get("LOG_FILE_PATH", "logs/liteagent.log"),
            file_max_bytes=int(os.environ.get("LOG_FILE_MAX_BYTES", str(10 * 1024 * 1024))),
            file_backup_count=int(os.environ.get("LOG_FILE_BACKUP_COUNT", "5")),
            file_rotation_type=os.environ.get("LOG_FILE_ROTATION_TYPE", "size"),
            include_request_body=os.environ.get("LOG_INCLUDE_REQUEST_BODY", "false").lower() == "true",
            timezone=os.environ.get("LOG_TIMEZONE", "UTC"),
        )


class JSONFormatter(logging.Formatter):
    """JSON log formatter with request ID and trace ID support."""

    def __init__(
        self,
        include_timestamp: bool = True,
        service_name: str = "liteagent",
    ):
        super().__init__()
        self.include_timestamp = include_timestamp
        self.service_name = service_name

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data: dict[str, Any] = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "service": self.service_name,
        }

        if self.include_timestamp:
            log_data["timestamp"] = datetime.now(timezone.utc).isoformat()

        # Add request ID from context
        request_id = _request_id.get()
        if request_id:
            log_data["request_id"] = request_id

        # Add trace ID from context
        trace_id = _trace_id.get()
        if trace_id:
            log_data["trace_id"] = trace_id

        # Add context from ContextVar
        context = _log_context.get()
        if context:
            log_data.update(context)

        # Add extra fields from record
        if hasattr(record, "extra_fields"):
            log_data.update(record.extra_fields)

        # Add source location for errors
        if record.levelno >= logging.ERROR:
            log_data["source"] = {
                "file": record.pathname,
                "line": record.lineno,
                "function": record.funcName,
            }

        # Handle exceptions
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else "Unknown",
                "message": str(record.exc_info[1]) if record.exc_info[1] else "",
                "stacktrace": self.formatException(record.exc_info),
            }

        return json.dumps(log_data, default=str)


class TextFormatter(logging.Formatter):
    """Human-readable text log formatter with request ID support."""

    def __init__(self, include_timestamp: bool = True):
        # Include request_id in format if present
        fmt = "%(levelname)s - %(name)s - %(message)s"
        if include_timestamp:
            fmt = "%(asctime)s - " + fmt
        super().__init__(fmt, datefmt="%Y-%m-%d %H:%M:%S")

    def format(self, record: logging.LogRecord) -> str:
        """Format with request ID if available."""
        # Add request_id to the record if available
        request_id = _request_id.get()
        if request_id:
            record.msg = f"[{request_id[:8]}] {record.msg}"
        return super().format(record)


# Global handlers to allow reconfiguration
_console_handler: logging.Handler | None = None
_file_handler: logging.Handler | None = None


def setup_logging(
    config: LogConfig,
    stream: IO[str] | None = None,
    service_name: str = "liteagent",
) -> None:
    """Set up logging with the given configuration."""
    global _console_handler, _file_handler

    root_logger = logging.getLogger("liteagent")

    # Remove existing handlers
    if _console_handler:
        root_logger.removeHandler(_console_handler)
    if _file_handler:
        root_logger.removeHandler(_file_handler)

    # Create formatter based on config
    if config.format == "json":
        formatter = JSONFormatter(
            include_timestamp=config.include_timestamp,
            service_name=service_name,
        )
    else:
        formatter = TextFormatter(include_timestamp=config.include_timestamp)

    # Create console handler
    _console_handler = logging.StreamHandler(stream or sys.stdout)
    _console_handler.setFormatter(formatter)
    root_logger.addHandler(_console_handler)

    # Create file handler if enabled
    if config.file_enabled:
        # Ensure log directory exists
        log_path = Path(config.file_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        if config.file_rotation_type == "time":
            _file_handler = TimedRotatingFileHandler(
                config.file_path,
                when=config.file_rotation_when,
                interval=config.file_rotation_interval,
                backupCount=config.file_backup_count,
                encoding="utf-8",
            )
        else:
            _file_handler = RotatingFileHandler(
                config.file_path,
                maxBytes=config.file_max_bytes,
                backupCount=config.file_backup_count,
                encoding="utf-8",
            )

        _file_handler.setFormatter(formatter)
        root_logger.addHandler(_file_handler)

    # Configure root logger
    root_logger.setLevel(getattr(logging, config.level))
    root_logger.propagate = False


def get_logger(name: str = "") -> logging.Logger:
    """Get a logger with the given name."""
    if name:
        return logging.getLogger(f"liteagent.{name}")
    return logging.getLogger("liteagent")


class LogContext:
    """Context manager for adding fields to all logs within a block."""

    def __init__(self, **fields: Any):
        self.fields = fields
        self.token = None

    def __enter__(self):
        current = _log_context.get()
        merged = {**current, **self.fields}
        self.token = _log_context.set(merged)
        return self

    def __exit__(self, *args):
        if self.token is not None:
            _log_context.reset(self.token)


class RequestLogger:
    """Logger for HTTP requests."""

    def __init__(self):
        self.logger = get_logger("http")

    def log_request_start(
        self,
        method: str,
        path: str,
        request_id: str,
        **extra: Any,
    ) -> None:
        """Log the start of an HTTP request."""
        self.logger.info(
            f"{method} {path} started",
            extra={
                "extra_fields": {
                    "event": "request_started",
                    "method": method,
                    "path": path,
                    "request_id": request_id,
                    **extra,
                }
            },
        )

    def log_request_end(
        self,
        method: str,
        path: str,
        status_code: int,
        duration_ms: float,
        request_id: str,
        **extra: Any,
    ) -> None:
        """Log the completion of an HTTP request."""
        self.logger.info(
            f"{method} {path} completed with {status_code}",
            extra={
                "extra_fields": {
                    "event": "request_completed",
                    "method": method,
                    "path": path,
                    "status_code": status_code,
                    "duration_ms": duration_ms,
                    "request_id": request_id,
                    **extra,
                }
            },
        )

    def log_request_error(
        self,
        method: str,
        path: str,
        error: str,
        request_id: str,
        **extra: Any,
    ) -> None:
        """Log an HTTP request error."""
        self.logger.error(
            f"{method} {path} error: {error}",
            extra={
                "extra_fields": {
                    "event": "request_error",
                    "method": method,
                    "path": path,
                    "error": error,
                    "request_id": request_id,
                    **extra,
                }
            },
        )


class LLMCallLogger:
    """Logger for LLM API calls."""

    def __init__(self):
        self.logger = get_logger("llm")

    def log_call_start(
        self,
        provider: str,
        model: str,
        agent_id: str | None = None,
        **extra: Any,
    ) -> None:
        """Log the start of an LLM call."""
        self.logger.info(
            f"LLM call to {provider}/{model} started",
            extra={
                "extra_fields": {
                    "event": "llm_call_started",
                    "provider": provider,
                    "model": model,
                    "agent_id": agent_id,
                    **extra,
                }
            },
        )

    def log_call_end(
        self,
        provider: str,
        model: str,
        duration_ms: float,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        total_tokens: int | None = None,
        agent_id: str | None = None,
        **extra: Any,
    ) -> None:
        """Log the completion of an LLM call."""
        self.logger.info(
            f"LLM call to {provider}/{model} completed in {duration_ms}ms",
            extra={
                "extra_fields": {
                    "event": "llm_call_completed",
                    "provider": provider,
                    "model": model,
                    "duration_ms": duration_ms,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "agent_id": agent_id,
                    **extra,
                }
            },
        )

    def log_call_error(
        self,
        provider: str,
        model: str,
        error: str,
        agent_id: str | None = None,
        **extra: Any,
    ) -> None:
        """Log an LLM call error."""
        self.logger.error(
            f"LLM call to {provider}/{model} failed: {error}",
            extra={
                "extra_fields": {
                    "event": "llm_call_error",
                    "provider": provider,
                    "model": model,
                    "error": error,
                    "agent_id": agent_id,
                    **extra,
                }
            },
        )


class MetricsCollector:
    """Simple in-memory metrics collector."""

    def __init__(self):
        self._counters: dict[str, int] = {}
        self._histograms: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def increment(
        self,
        name: str,
        value: int = 1,
        labels: dict[str, str] | None = None,
    ) -> None:
        """Increment a counter metric."""
        key = self._make_key(name, labels)
        with self._lock:
            self._counters[key] = self._counters.get(key, 0) + value

    def get_counter(self, name: str, labels: dict[str, str] | None = None) -> int:
        """Get the current value of a counter."""
        key = self._make_key(name, labels)
        with self._lock:
            return self._counters.get(key, 0)

    def record_histogram(
        self,
        name: str,
        value: float,
        labels: dict[str, str] | None = None,
    ) -> None:
        """Record a value in a histogram metric."""
        key = self._make_key(name, labels)
        with self._lock:
            if key not in self._histograms:
                self._histograms[key] = []
            self._histograms[key].append(value)

    def get_histogram(
        self, name: str, labels: dict[str, str] | None = None
    ) -> list[float] | None:
        """Get histogram values."""
        key = self._make_key(name, labels)
        with self._lock:
            return self._histograms.get(key)

    def _make_key(self, name: str, labels: dict[str, str] | None) -> str:
        """Create a unique key for a metric with labels."""
        if not labels:
            return name
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"

    def reset(self) -> None:
        """Reset all metrics (for testing)."""
        with self._lock:
            self._counters.clear()
            self._histograms.clear()


# Global metrics instance
metrics = MetricsCollector()


# Request ID helper functions
def set_request_id(request_id: str) -> None:
    """Set the current request ID."""
    _request_id.set(request_id)


def get_request_id() -> str | None:
    """Get the current request ID."""
    return _request_id.get()


def generate_request_id() -> str:
    """Generate a new request ID."""
    return str(uuid4())


def set_trace_id(trace_id: str) -> None:
    """Set the current trace ID."""
    _trace_id.set(trace_id)


def get_trace_id() -> str | None:
    """Get the current trace ID."""
    return _trace_id.get()


class RequestIDContext:
    """Context manager for request ID."""

    def __init__(self, request_id: str | None = None):
        self.request_id = request_id or generate_request_id()
        self._token = None

    def __enter__(self) -> str:
        self._token = _request_id.set(self.request_id)
        return self.request_id

    def __exit__(self, *args: Any) -> None:
        if self._token:
            _request_id.reset(self._token)


class WorkflowLogger:
    """Logger for workflow execution."""

    def __init__(self):
        self.logger = get_logger("workflow")

    def log_workflow_start(
        self,
        workflow_id: str,
        workflow_name: str,
        **extra: Any,
    ) -> None:
        """Log workflow execution start."""
        self.logger.info(
            f"Workflow {workflow_name} started",
            extra={
                "extra_fields": {
                    "event": "workflow_started",
                    "workflow_id": workflow_id,
                    "workflow_name": workflow_name,
                    **extra,
                }
            },
        )

    def log_workflow_end(
        self,
        workflow_id: str,
        workflow_name: str,
        status: str,
        duration_ms: float,
        **extra: Any,
    ) -> None:
        """Log workflow execution completion."""
        self.logger.info(
            f"Workflow {workflow_name} completed with status {status}",
            extra={
                "extra_fields": {
                    "event": "workflow_completed",
                    "workflow_id": workflow_id,
                    "workflow_name": workflow_name,
                    "status": status,
                    "duration_ms": duration_ms,
                    **extra,
                }
            },
        )

    def log_node_execution(
        self,
        workflow_id: str,
        node_id: str,
        node_type: str,
        status: str,
        duration_ms: float,
        **extra: Any,
    ) -> None:
        """Log node execution."""
        self.logger.debug(
            f"Node {node_id} ({node_type}) completed with {status}",
            extra={
                "extra_fields": {
                    "event": "node_executed",
                    "workflow_id": workflow_id,
                    "node_id": node_id,
                    "node_type": node_type,
                    "status": status,
                    "duration_ms": duration_ms,
                    **extra,
                }
            },
        )


class RetrievalLogger:
    """Logger for RAG retrieval operations."""

    def __init__(self):
        self.logger = get_logger("retrieval")

    def log_retrieval_start(
        self,
        query: str,
        datasource_ids: list[str],
        **extra: Any,
    ) -> None:
        """Log retrieval start."""
        self.logger.info(
            f"Retrieval started for query",
            extra={
                "extra_fields": {
                    "event": "retrieval_started",
                    "query_length": len(query),
                    "datasource_count": len(datasource_ids),
                    **extra,
                }
            },
        )

    def log_retrieval_end(
        self,
        query: str,
        result_count: int,
        duration_ms: float,
        **extra: Any,
    ) -> None:
        """Log retrieval completion."""
        self.logger.info(
            f"Retrieval completed with {result_count} results",
            extra={
                "extra_fields": {
                    "event": "retrieval_completed",
                    "query_length": len(query),
                    "result_count": result_count,
                    "duration_ms": duration_ms,
                    **extra,
                }
            },
        )


class AgentLogger:
    """Logger for agent execution."""

    def __init__(self):
        self.logger = get_logger("agent")

    def log_agent_start(
        self,
        agent_id: str,
        agent_name: str,
        **extra: Any,
    ) -> None:
        """Log agent execution start."""
        self.logger.info(
            f"Agent {agent_name} started",
            extra={
                "extra_fields": {
                    "event": "agent_started",
                    "agent_id": agent_id,
                    "agent_name": agent_name,
                    **extra,
                }
            },
        )

    def log_agent_step(
        self,
        agent_id: str,
        step_number: int,
        action: str,
        **extra: Any,
    ) -> None:
        """Log agent step."""
        self.logger.debug(
            f"Agent step {step_number}: {action}",
            extra={
                "extra_fields": {
                    "event": "agent_step",
                    "agent_id": agent_id,
                    "step_number": step_number,
                    "action": action,
                    **extra,
                }
            },
        )

    def log_agent_end(
        self,
        agent_id: str,
        agent_name: str,
        status: str,
        steps_count: int,
        duration_ms: float,
        **extra: Any,
    ) -> None:
        """Log agent execution completion."""
        self.logger.info(
            f"Agent {agent_name} completed with status {status}",
            extra={
                "extra_fields": {
                    "event": "agent_completed",
                    "agent_id": agent_id,
                    "agent_name": agent_name,
                    "status": status,
                    "steps_count": steps_count,
                    "duration_ms": duration_ms,
                    **extra,
                }
            },
        )

    def log_tool_call(
        self,
        agent_id: str,
        tool_name: str,
        duration_ms: float,
        success: bool,
        **extra: Any,
    ) -> None:
        """Log tool call."""
        self.logger.debug(
            f"Tool {tool_name} {'succeeded' if success else 'failed'}",
            extra={
                "extra_fields": {
                    "event": "tool_call",
                    "agent_id": agent_id,
                    "tool_name": tool_name,
                    "duration_ms": duration_ms,
                    "success": success,
                    **extra,
                }
            },
        )


# Create global logger instances
request_logger = RequestLogger()
llm_logger = LLMCallLogger()
workflow_logger = WorkflowLogger()
retrieval_logger = RetrievalLogger()
agent_logger = AgentLogger()
