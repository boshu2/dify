"""
Structured logging and metrics collection for LiteAgent.
Provides JSON-formatted logs for production and text logs for development.
"""
import os
import sys
import json
import logging
import threading
from datetime import datetime, timezone
from typing import Any, IO
from dataclasses import dataclass, field
from contextvars import ContextVar


# Context variable for log context
_log_context: ContextVar[dict[str, Any]] = ContextVar("log_context", default={})


@dataclass
class LogConfig:
    """Configuration for logging."""

    level: str = "INFO"
    format: str = "json"  # "json" or "text"
    include_timestamp: bool = True

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
        )


class JSONFormatter(logging.Formatter):
    """JSON log formatter."""

    def __init__(self, include_timestamp: bool = True):
        super().__init__()
        self.include_timestamp = include_timestamp

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        if self.include_timestamp:
            log_data["timestamp"] = datetime.now(timezone.utc).isoformat()

        # Add context from ContextVar
        context = _log_context.get()
        if context:
            log_data.update(context)

        # Add extra fields from record
        if hasattr(record, "extra_fields"):
            log_data.update(record.extra_fields)

        # Handle exceptions
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data)


class TextFormatter(logging.Formatter):
    """Human-readable text log formatter."""

    def __init__(self, include_timestamp: bool = True):
        fmt = "%(levelname)s - %(name)s - %(message)s"
        if include_timestamp:
            fmt = "%(asctime)s - " + fmt
        super().__init__(fmt, datefmt="%Y-%m-%d %H:%M:%S")


# Global handler to allow reconfiguration
_handler: logging.Handler | None = None


def setup_logging(config: LogConfig, stream: IO[str] | None = None) -> None:
    """Set up logging with the given configuration."""
    global _handler

    # Remove existing handler if any
    root_logger = logging.getLogger("liteagent")
    if _handler:
        root_logger.removeHandler(_handler)

    # Create new handler
    _handler = logging.StreamHandler(stream or sys.stdout)

    # Set formatter based on config
    if config.format == "json":
        _handler.setFormatter(JSONFormatter(include_timestamp=config.include_timestamp))
    else:
        _handler.setFormatter(TextFormatter(include_timestamp=config.include_timestamp))

    # Configure root logger
    root_logger.setLevel(getattr(logging, config.level))
    root_logger.addHandler(_handler)
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
