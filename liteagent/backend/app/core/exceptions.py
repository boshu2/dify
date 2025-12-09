"""
Custom exceptions for LiteAgent.
Provides domain-specific error handling with proper HTTP status codes.
"""
from typing import Any

from fastapi import HTTPException


class LiteAgentException(Exception):
    """Base exception for all LiteAgent errors."""

    status_code: int = 500
    error_code: str = "INTERNAL_ERROR"

    def __init__(
        self,
        message: str,
        details: dict[str, Any] | None = None,
    ):
        self.message = message
        self.details = details
        super().__init__(message)

    def __str__(self) -> str:
        return self.message

    def to_http_exception(self) -> HTTPException:
        """Convert to FastAPI HTTPException."""
        detail = {
            "message": self.message,
            "error_code": self.error_code,
        }
        if self.details:
            detail["details"] = self.details
        return HTTPException(status_code=self.status_code, detail=detail)


# ============== Provider Exceptions ==============


class ProviderNotFoundError(LiteAgentException):
    """Raised when a provider is not found."""

    status_code = 404
    error_code = "PROVIDER_NOT_FOUND"

    def __init__(self, provider_id: str, details: dict[str, Any] | None = None):
        super().__init__(f"Provider not found: {provider_id}", details)
        self.provider_id = provider_id


class ProviderConfigurationError(LiteAgentException):
    """Raised when provider configuration is invalid."""

    status_code = 400
    error_code = "PROVIDER_CONFIG_ERROR"


# ============== DataSource Exceptions ==============


class DataSourceNotFoundError(LiteAgentException):
    """Raised when a data source is not found."""

    status_code = 404
    error_code = "DATASOURCE_NOT_FOUND"

    def __init__(self, datasource_id: str, details: dict[str, Any] | None = None):
        super().__init__(f"Data source not found: {datasource_id}", details)
        self.datasource_id = datasource_id


class DataSourceFetchError(LiteAgentException):
    """Raised when fetching data from a source fails."""

    status_code = 502
    error_code = "DATASOURCE_FETCH_ERROR"


# ============== Agent Exceptions ==============


class AgentNotFoundError(LiteAgentException):
    """Raised when an agent is not found."""

    status_code = 404
    error_code = "AGENT_NOT_FOUND"

    def __init__(self, agent_id: str, details: dict[str, Any] | None = None):
        super().__init__(f"Agent not found: {agent_id}", details)
        self.agent_id = agent_id


class AgentExecutionError(LiteAgentException):
    """Raised when agent execution fails."""

    status_code = 500
    error_code = "AGENT_EXECUTION_ERROR"


# ============== Security Exceptions ==============


class AuthenticationError(LiteAgentException):
    """Raised when authentication fails."""

    status_code = 401
    error_code = "AUTHENTICATION_ERROR"


class RateLimitExceededError(LiteAgentException):
    """Raised when rate limit is exceeded."""

    status_code = 429
    error_code = "RATE_LIMIT_EXCEEDED"

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        retry_after: int = 60,
        details: dict[str, Any] | None = None,
    ):
        super().__init__(message, details)
        self.retry_after = retry_after


# ============== Validation Exceptions ==============


class ValidationError(LiteAgentException):
    """Raised when validation fails."""

    status_code = 422
    error_code = "VALIDATION_ERROR"

    def __init__(
        self,
        message: str,
        field: str | None = None,
        details: dict[str, Any] | None = None,
    ):
        super().__init__(message, details)
        self.field = field
