"""
Unit tests for logging and monitoring.
Tests structured logging, request tracking, and metrics collection.
"""
import json
import pytest
import logging
from io import StringIO
from unittest.mock import Mock, patch

from app.core.logging import (
    LogConfig,
    get_logger,
    setup_logging,
    LogContext,
    RequestLogger,
    LLMCallLogger,
)


class TestLogConfig:
    """Tests for logging configuration."""

    def test_default_config(self):
        """Test default log configuration."""
        config = LogConfig()
        assert config.level == "INFO"
        assert config.format == "json"
        assert config.include_timestamp is True

    def test_config_from_env(self):
        """Test configuration from environment variables."""
        with patch.dict(
            "os.environ",
            {
                "LOG_LEVEL": "DEBUG",
                "LOG_FORMAT": "text",
            },
        ):
            config = LogConfig.from_env()
            assert config.level == "DEBUG"
            assert config.format == "text"

    def test_config_validates_level(self):
        """Test that invalid log levels are rejected."""
        with pytest.raises(ValueError):
            LogConfig(level="INVALID")


class TestGetLogger:
    """Tests for logger factory function."""

    def test_get_logger_by_name(self):
        """Test getting a named logger."""
        logger = get_logger("test_module")
        assert logger.name == "liteagent.test_module"

    def test_get_logger_with_default_name(self):
        """Test getting logger with default name."""
        logger = get_logger()
        assert "liteagent" in logger.name


class TestSetupLogging:
    """Tests for logging setup."""

    def test_setup_json_logging(self):
        """Test JSON-formatted logging setup."""
        stream = StringIO()
        config = LogConfig(level="DEBUG", format="json")
        setup_logging(config, stream=stream)

        logger = get_logger("test")
        logger.info("Test message", extra={"custom_field": "value"})

        stream.seek(0)
        output = stream.read()
        log_data = json.loads(output.strip())

        assert log_data["message"] == "Test message"
        assert log_data["level"] == "INFO"
        assert "timestamp" in log_data

    def test_setup_text_logging(self):
        """Test text-formatted logging setup."""
        stream = StringIO()
        config = LogConfig(level="DEBUG", format="text")
        setup_logging(config, stream=stream)

        logger = get_logger("test")
        logger.info("Test message")

        stream.seek(0)
        output = stream.read()
        assert "Test message" in output
        assert "INFO" in output


class TestLogContext:
    """Tests for logging context management."""

    def test_log_context_adds_fields(self):
        """Test that context adds fields to all log messages."""
        stream = StringIO()
        setup_logging(LogConfig(format="json"), stream=stream)

        with LogContext(request_id="req-123", user_id="user-456"):
            logger = get_logger("test")
            logger.info("Contextual message")

        stream.seek(0)
        output = stream.read()
        log_data = json.loads(output.strip())

        assert log_data["request_id"] == "req-123"
        assert log_data["user_id"] == "user-456"

    def test_log_context_nesting(self):
        """Test nested log contexts."""
        stream = StringIO()
        setup_logging(LogConfig(format="json"), stream=stream)

        with LogContext(outer="outer_value"):
            with LogContext(inner="inner_value"):
                logger = get_logger("test")
                logger.info("Nested message")

        stream.seek(0)
        output = stream.read()
        log_data = json.loads(output.strip())

        assert log_data["outer"] == "outer_value"
        assert log_data["inner"] == "inner_value"

    def test_log_context_cleanup(self):
        """Test context is cleaned up after exiting."""
        stream = StringIO()
        setup_logging(LogConfig(format="json"), stream=stream)

        with LogContext(temp="temporary"):
            pass

        logger = get_logger("test")
        logger.info("After context")

        stream.seek(0)
        output = stream.read()
        log_data = json.loads(output.strip())

        assert "temp" not in log_data


class TestRequestLogger:
    """Tests for HTTP request logging."""

    def test_log_request_start(self):
        """Test logging request start."""
        stream = StringIO()
        setup_logging(LogConfig(format="json"), stream=stream)

        request_logger = RequestLogger()
        request_logger.log_request_start(
            method="GET",
            path="/api/agents",
            request_id="req-123",
        )

        stream.seek(0)
        output = stream.read()
        log_data = json.loads(output.strip())

        assert log_data["event"] == "request_started"
        assert log_data["method"] == "GET"
        assert log_data["path"] == "/api/agents"

    def test_log_request_end(self):
        """Test logging request completion."""
        stream = StringIO()
        setup_logging(LogConfig(format="json"), stream=stream)

        request_logger = RequestLogger()
        request_logger.log_request_end(
            method="GET",
            path="/api/agents",
            status_code=200,
            duration_ms=45.5,
            request_id="req-123",
        )

        stream.seek(0)
        output = stream.read()
        log_data = json.loads(output.strip())

        assert log_data["event"] == "request_completed"
        assert log_data["status_code"] == 200
        assert log_data["duration_ms"] == 45.5

    def test_log_request_error(self):
        """Test logging request error."""
        stream = StringIO()
        setup_logging(LogConfig(format="json"), stream=stream)

        request_logger = RequestLogger()
        request_logger.log_request_error(
            method="POST",
            path="/api/agents/chat",
            error="Internal Server Error",
            request_id="req-123",
        )

        stream.seek(0)
        output = stream.read()
        log_data = json.loads(output.strip())

        assert log_data["event"] == "request_error"
        assert log_data["error"] == "Internal Server Error"


class TestLLMCallLogger:
    """Tests for LLM call logging."""

    def test_log_llm_call_start(self):
        """Test logging LLM call start."""
        stream = StringIO()
        setup_logging(LogConfig(format="json"), stream=stream)

        llm_logger = LLMCallLogger()
        llm_logger.log_call_start(
            provider="openai",
            model="gpt-4o",
            agent_id="agent-123",
        )

        stream.seek(0)
        output = stream.read()
        log_data = json.loads(output.strip())

        assert log_data["event"] == "llm_call_started"
        assert log_data["provider"] == "openai"
        assert log_data["model"] == "gpt-4o"

    def test_log_llm_call_end(self):
        """Test logging LLM call completion with token counts."""
        stream = StringIO()
        setup_logging(LogConfig(format="json"), stream=stream)

        llm_logger = LLMCallLogger()
        llm_logger.log_call_end(
            provider="openai",
            model="gpt-4o",
            duration_ms=1234.5,
            prompt_tokens=150,
            completion_tokens=250,
            total_tokens=400,
            agent_id="agent-123",
        )

        stream.seek(0)
        output = stream.read()
        log_data = json.loads(output.strip())

        assert log_data["event"] == "llm_call_completed"
        assert log_data["duration_ms"] == 1234.5
        assert log_data["prompt_tokens"] == 150
        assert log_data["completion_tokens"] == 250
        assert log_data["total_tokens"] == 400

    def test_log_llm_call_error(self):
        """Test logging LLM call error."""
        stream = StringIO()
        setup_logging(LogConfig(format="json"), stream=stream)

        llm_logger = LLMCallLogger()
        llm_logger.log_call_error(
            provider="openai",
            model="gpt-4o",
            error="Rate limit exceeded",
            agent_id="agent-123",
        )

        stream.seek(0)
        output = stream.read()
        log_data = json.loads(output.strip())

        assert log_data["event"] == "llm_call_error"
        assert log_data["error"] == "Rate limit exceeded"


class TestMetricsCollection:
    """Tests for metrics collection."""

    def test_increment_request_counter(self):
        """Test incrementing request counter."""
        from app.core.logging import metrics

        labels = {"method": "GET", "path": "/api/agents"}
        initial = metrics.get_counter("requests_total", labels=labels)
        metrics.increment("requests_total", labels=labels)
        assert metrics.get_counter("requests_total", labels=labels) == initial + 1

    def test_record_request_duration(self):
        """Test recording request duration histogram."""
        from app.core.logging import metrics

        labels = {"method": "GET", "path": "/api/agents"}
        metrics.record_histogram(
            "request_duration_seconds",
            0.045,
            labels=labels,
        )
        histogram = metrics.get_histogram("request_duration_seconds", labels=labels)
        assert histogram is not None
        assert 0.045 in histogram

    def test_record_llm_tokens(self):
        """Test recording LLM token usage."""
        from app.core.logging import metrics

        labels = {"provider": "openai", "model": "gpt-4o"}
        initial = metrics.get_counter("llm_tokens_total", labels=labels)
        metrics.increment(
            "llm_tokens_total",
            value=400,
            labels=labels,
        )
        tokens = metrics.get_counter("llm_tokens_total", labels=labels)
        assert tokens >= initial + 400
