"""
Unit tests for custom exceptions and error handling.
"""
import pytest
from fastapi import HTTPException

from app.core.exceptions import (
    LiteAgentException,
    ProviderNotFoundError,
    ProviderConfigurationError,
    DataSourceNotFoundError,
    DataSourceFetchError,
    AgentNotFoundError,
    AgentExecutionError,
    AuthenticationError,
    RateLimitExceededError,
    ValidationError,
)


class TestLiteAgentException:
    """Tests for base exception class."""

    def test_create_exception(self):
        """Test creating base exception."""
        exc = LiteAgentException("Something went wrong")
        assert str(exc) == "Something went wrong"
        assert exc.status_code == 500
        assert exc.error_code == "INTERNAL_ERROR"

    def test_exception_with_details(self):
        """Test exception with additional details."""
        exc = LiteAgentException(
            "Something went wrong",
            details={"key": "value"},
        )
        assert exc.details == {"key": "value"}


class TestProviderExceptions:
    """Tests for provider-related exceptions."""

    def test_provider_not_found(self):
        """Test ProviderNotFoundError."""
        exc = ProviderNotFoundError("provider-123")
        assert "provider-123" in str(exc)
        assert exc.status_code == 404
        assert exc.error_code == "PROVIDER_NOT_FOUND"

    def test_provider_configuration_error(self):
        """Test ProviderConfigurationError."""
        exc = ProviderConfigurationError("Invalid API key")
        assert "Invalid API key" in str(exc)
        assert exc.status_code == 400
        assert exc.error_code == "PROVIDER_CONFIG_ERROR"


class TestDataSourceExceptions:
    """Tests for datasource-related exceptions."""

    def test_datasource_not_found(self):
        """Test DataSourceNotFoundError."""
        exc = DataSourceNotFoundError("ds-123")
        assert "ds-123" in str(exc)
        assert exc.status_code == 404
        assert exc.error_code == "DATASOURCE_NOT_FOUND"

    def test_datasource_fetch_error(self):
        """Test DataSourceFetchError."""
        exc = DataSourceFetchError("Failed to fetch URL")
        assert "Failed to fetch URL" in str(exc)
        assert exc.status_code == 502
        assert exc.error_code == "DATASOURCE_FETCH_ERROR"


class TestAgentExceptions:
    """Tests for agent-related exceptions."""

    def test_agent_not_found(self):
        """Test AgentNotFoundError."""
        exc = AgentNotFoundError("agent-123")
        assert "agent-123" in str(exc)
        assert exc.status_code == 404
        assert exc.error_code == "AGENT_NOT_FOUND"

    def test_agent_execution_error(self):
        """Test AgentExecutionError."""
        exc = AgentExecutionError("LLM API failed")
        assert "LLM API failed" in str(exc)
        assert exc.status_code == 500
        assert exc.error_code == "AGENT_EXECUTION_ERROR"


class TestSecurityExceptions:
    """Tests for security-related exceptions."""

    def test_authentication_error(self):
        """Test AuthenticationError."""
        exc = AuthenticationError("Invalid token")
        assert "Invalid token" in str(exc)
        assert exc.status_code == 401
        assert exc.error_code == "AUTHENTICATION_ERROR"

    def test_rate_limit_exceeded(self):
        """Test RateLimitExceededError."""
        exc = RateLimitExceededError("Too many requests")
        assert "Too many requests" in str(exc)
        assert exc.status_code == 429
        assert exc.error_code == "RATE_LIMIT_EXCEEDED"
        assert exc.retry_after is not None


class TestValidationException:
    """Tests for validation exceptions."""

    def test_validation_error(self):
        """Test ValidationError."""
        exc = ValidationError("Invalid input", field="email")
        assert "Invalid input" in str(exc)
        assert exc.status_code == 422
        assert exc.error_code == "VALIDATION_ERROR"
        assert exc.field == "email"


class TestExceptionToHTTPException:
    """Tests for converting custom exceptions to HTTPException."""

    def test_convert_to_http_exception(self):
        """Test converting LiteAgentException to HTTPException."""
        exc = ProviderNotFoundError("provider-123")
        http_exc = exc.to_http_exception()

        assert isinstance(http_exc, HTTPException)
        assert http_exc.status_code == 404
        assert "provider-123" in http_exc.detail["message"]
        assert http_exc.detail["error_code"] == "PROVIDER_NOT_FOUND"

    def test_http_exception_includes_details(self):
        """Test that HTTP exception includes details."""
        exc = LiteAgentException(
            "Error",
            details={"extra": "info"},
        )
        http_exc = exc.to_http_exception()

        assert http_exc.detail["details"] == {"extra": "info"}
