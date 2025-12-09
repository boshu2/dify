"""
Input validation and sanitization for LLM requests.
Provides security-focused validation to prevent injection attacks.
"""
import html
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ValidationErrorType(str, Enum):
    """Types of validation errors."""

    REQUIRED = "required"
    TYPE_ERROR = "type_error"
    LENGTH_ERROR = "length_error"
    FORMAT_ERROR = "format_error"
    CONTENT_ERROR = "content_error"
    INJECTION_DETECTED = "injection_detected"
    FORBIDDEN_CONTENT = "forbidden_content"


@dataclass
class ValidationError:
    """A validation error."""

    field: str
    message: str
    error_type: ValidationErrorType
    value: Any = None


@dataclass
class ValidationResult:
    """Result of validation."""

    is_valid: bool
    errors: list[ValidationError] = field(default_factory=list)
    sanitized_value: Any = None
    warnings: list[str] = field(default_factory=list)

    @classmethod
    def success(cls, value: Any = None) -> "ValidationResult":
        """Create a successful result."""
        return cls(is_valid=True, sanitized_value=value)

    @classmethod
    def failure(cls, error: ValidationError) -> "ValidationResult":
        """Create a failure result."""
        return cls(is_valid=False, errors=[error])

    def add_error(self, error: ValidationError) -> None:
        """Add an error."""
        self.errors.append(error)
        self.is_valid = False

    def add_warning(self, warning: str) -> None:
        """Add a warning (doesn't fail validation)."""
        self.warnings.append(warning)


class ContentValidator:
    """Validates and sanitizes content."""

    # Patterns for detecting potential prompt injection
    INJECTION_PATTERNS = [
        r"ignore\s+(previous|all|above)\s+instructions?",
        r"disregard\s+(previous|all|above)\s+instructions?",
        r"forget\s+(everything|all)\s+(you\s+)?know",
        r"you\s+are\s+now\s+(?:a|an)\s+",
        r"act\s+as\s+(?:a|an)\s+",
        r"pretend\s+(?:to\s+be|you\s+are)\s+",
        r"new\s+instructions?:\s*",
        r"system\s*:\s*",
        r"\[INST\]",
        r"<\|im_start\|>",
        r"###\s*(?:System|Human|Assistant)\s*:",
    ]

    # Patterns for dangerous content
    DANGEROUS_PATTERNS = [
        r"<script[^>]*>",  # Script tags
        r"javascript:",  # JavaScript protocol
        r"on\w+\s*=",  # Event handlers
        r"data:\s*text/html",  # Data URLs with HTML
    ]

    def __init__(
        self,
        max_length: int = 100000,
        allow_html: bool = False,
        check_injection: bool = True,
        strip_whitespace: bool = True,
    ):
        """
        Initialize content validator.

        Args:
            max_length: Maximum content length.
            allow_html: Whether to allow HTML tags.
            check_injection: Whether to check for prompt injection.
            strip_whitespace: Whether to strip leading/trailing whitespace.
        """
        self.max_length = max_length
        self.allow_html = allow_html
        self.check_injection = check_injection
        self.strip_whitespace = strip_whitespace

        # Compile patterns
        self._injection_regex = [
            re.compile(p, re.IGNORECASE) for p in self.INJECTION_PATTERNS
        ]
        self._dangerous_regex = [
            re.compile(p, re.IGNORECASE) for p in self.DANGEROUS_PATTERNS
        ]

    def validate(self, content: str, field_name: str = "content") -> ValidationResult:
        """
        Validate and sanitize content.

        Args:
            content: Content to validate.
            field_name: Name of the field (for error messages).

        Returns:
            ValidationResult with sanitized content.
        """
        if content is None:
            return ValidationResult.failure(
                ValidationError(
                    field=field_name,
                    message="Content is required",
                    error_type=ValidationErrorType.REQUIRED,
                )
            )

        if not isinstance(content, str):
            return ValidationResult.failure(
                ValidationError(
                    field=field_name,
                    message="Content must be a string",
                    error_type=ValidationErrorType.TYPE_ERROR,
                    value=type(content).__name__,
                )
            )

        # Sanitize
        sanitized = content
        if self.strip_whitespace:
            sanitized = sanitized.strip()

        # Check length
        if len(sanitized) > self.max_length:
            return ValidationResult.failure(
                ValidationError(
                    field=field_name,
                    message=f"Content exceeds maximum length of {self.max_length}",
                    error_type=ValidationErrorType.LENGTH_ERROR,
                    value=len(sanitized),
                )
            )

        result = ValidationResult(is_valid=True, sanitized_value=sanitized)

        # Check for injection attempts
        if self.check_injection:
            injection_result = self._check_injection(sanitized, field_name)
            if not injection_result.is_valid:
                return injection_result
            result.warnings.extend(injection_result.warnings)

        # Check for dangerous content
        dangerous_result = self._check_dangerous(sanitized, field_name)
        if not dangerous_result.is_valid:
            return dangerous_result

        # Escape HTML if not allowed
        if not self.allow_html:
            sanitized = html.escape(sanitized)
            result.sanitized_value = sanitized

        return result

    def _check_injection(self, content: str, field_name: str) -> ValidationResult:
        """Check for prompt injection patterns."""
        result = ValidationResult(is_valid=True)

        for pattern in self._injection_regex:
            if pattern.search(content):
                result.add_warning(
                    f"Potential prompt injection pattern detected in {field_name}"
                )
                # Don't fail, just warn - let the application decide
                break

        return result

    def _check_dangerous(self, content: str, field_name: str) -> ValidationResult:
        """Check for dangerous content patterns."""
        for pattern in self._dangerous_regex:
            if pattern.search(content):
                return ValidationResult.failure(
                    ValidationError(
                        field=field_name,
                        message="Content contains potentially dangerous patterns",
                        error_type=ValidationErrorType.FORBIDDEN_CONTENT,
                    )
                )
        return ValidationResult(is_valid=True)


class MessageValidator:
    """Validates chat messages."""

    VALID_ROLES = {"system", "user", "assistant", "tool"}

    def __init__(
        self,
        max_message_length: int = 100000,
        max_messages: int = 1000,
        content_validator: ContentValidator | None = None,
    ):
        """
        Initialize message validator.

        Args:
            max_message_length: Maximum length per message.
            max_messages: Maximum number of messages.
            content_validator: Optional content validator.
        """
        self.max_message_length = max_message_length
        self.max_messages = max_messages
        self.content_validator = content_validator or ContentValidator(
            max_length=max_message_length
        )

    def validate_message(
        self,
        message: dict[str, Any],
        index: int = 0,
    ) -> ValidationResult:
        """
        Validate a single message.

        Args:
            message: Message dictionary.
            index: Message index (for error messages).

        Returns:
            ValidationResult.
        """
        if not isinstance(message, dict):
            return ValidationResult.failure(
                ValidationError(
                    field=f"messages[{index}]",
                    message="Message must be a dictionary",
                    error_type=ValidationErrorType.TYPE_ERROR,
                )
            )

        # Check role
        role = message.get("role")
        if not role:
            return ValidationResult.failure(
                ValidationError(
                    field=f"messages[{index}].role",
                    message="Message role is required",
                    error_type=ValidationErrorType.REQUIRED,
                )
            )

        if role not in self.VALID_ROLES:
            return ValidationResult.failure(
                ValidationError(
                    field=f"messages[{index}].role",
                    message=f"Invalid role: {role}. Must be one of {self.VALID_ROLES}",
                    error_type=ValidationErrorType.FORMAT_ERROR,
                    value=role,
                )
            )

        # Check content
        content = message.get("content")
        if content is not None:
            content_result = self.content_validator.validate(
                content,
                field_name=f"messages[{index}].content",
            )
            if not content_result.is_valid:
                return content_result

            # Return sanitized message
            sanitized_message = message.copy()
            sanitized_message["content"] = content_result.sanitized_value
            return ValidationResult(
                is_valid=True,
                sanitized_value=sanitized_message,
                warnings=content_result.warnings,
            )

        return ValidationResult(is_valid=True, sanitized_value=message)

    def validate_messages(
        self,
        messages: list[dict[str, Any]],
    ) -> ValidationResult:
        """
        Validate a list of messages.

        Args:
            messages: List of messages.

        Returns:
            ValidationResult with sanitized messages.
        """
        if not isinstance(messages, list):
            return ValidationResult.failure(
                ValidationError(
                    field="messages",
                    message="Messages must be a list",
                    error_type=ValidationErrorType.TYPE_ERROR,
                )
            )

        if len(messages) > self.max_messages:
            return ValidationResult.failure(
                ValidationError(
                    field="messages",
                    message=f"Too many messages: {len(messages)} > {self.max_messages}",
                    error_type=ValidationErrorType.LENGTH_ERROR,
                    value=len(messages),
                )
            )

        sanitized_messages = []
        all_warnings = []

        for i, message in enumerate(messages):
            result = self.validate_message(message, i)
            if not result.is_valid:
                return result
            sanitized_messages.append(result.sanitized_value)
            all_warnings.extend(result.warnings)

        return ValidationResult(
            is_valid=True,
            sanitized_value=sanitized_messages,
            warnings=all_warnings,
        )


class RequestValidator:
    """Validates complete API requests."""

    def __init__(
        self,
        message_validator: MessageValidator | None = None,
        allowed_models: list[str] | None = None,
        max_tokens_limit: int = 128000,
    ):
        """
        Initialize request validator.

        Args:
            message_validator: Optional message validator.
            allowed_models: List of allowed model names.
            max_tokens_limit: Maximum tokens allowed.
        """
        self.message_validator = message_validator or MessageValidator()
        self.allowed_models = allowed_models
        self.max_tokens_limit = max_tokens_limit

    def validate_chat_request(
        self,
        request: dict[str, Any],
    ) -> ValidationResult:
        """
        Validate a chat completion request.

        Args:
            request: Request dictionary.

        Returns:
            ValidationResult with sanitized request.
        """
        if not isinstance(request, dict):
            return ValidationResult.failure(
                ValidationError(
                    field="request",
                    message="Request must be a dictionary",
                    error_type=ValidationErrorType.TYPE_ERROR,
                )
            )

        sanitized = request.copy()
        all_warnings = []

        # Validate messages
        messages = request.get("messages")
        if messages is None:
            return ValidationResult.failure(
                ValidationError(
                    field="messages",
                    message="Messages are required",
                    error_type=ValidationErrorType.REQUIRED,
                )
            )

        messages_result = self.message_validator.validate_messages(messages)
        if not messages_result.is_valid:
            return messages_result

        sanitized["messages"] = messages_result.sanitized_value
        all_warnings.extend(messages_result.warnings)

        # Validate model
        model = request.get("model")
        if self.allowed_models and model and model not in self.allowed_models:
            return ValidationResult.failure(
                ValidationError(
                    field="model",
                    message=f"Model '{model}' is not allowed",
                    error_type=ValidationErrorType.FORMAT_ERROR,
                    value=model,
                )
            )

        # Validate max_tokens
        max_tokens = request.get("max_tokens")
        if max_tokens is not None:
            if not isinstance(max_tokens, int) or max_tokens < 1:
                return ValidationResult.failure(
                    ValidationError(
                        field="max_tokens",
                        message="max_tokens must be a positive integer",
                        error_type=ValidationErrorType.TYPE_ERROR,
                        value=max_tokens,
                    )
                )
            if max_tokens > self.max_tokens_limit:
                return ValidationResult.failure(
                    ValidationError(
                        field="max_tokens",
                        message=f"max_tokens exceeds limit of {self.max_tokens_limit}",
                        error_type=ValidationErrorType.LENGTH_ERROR,
                        value=max_tokens,
                    )
                )

        # Validate temperature
        temperature = request.get("temperature")
        if temperature is not None:
            if not isinstance(temperature, (int, float)):
                return ValidationResult.failure(
                    ValidationError(
                        field="temperature",
                        message="temperature must be a number",
                        error_type=ValidationErrorType.TYPE_ERROR,
                        value=temperature,
                    )
                )
            if not 0 <= temperature <= 2:
                return ValidationResult.failure(
                    ValidationError(
                        field="temperature",
                        message="temperature must be between 0 and 2",
                        error_type=ValidationErrorType.FORMAT_ERROR,
                        value=temperature,
                    )
                )

        return ValidationResult(
            is_valid=True,
            sanitized_value=sanitized,
            warnings=all_warnings,
        )


# Utility functions
def sanitize_string(text: str, max_length: int = 10000) -> str:
    """
    Sanitize a string for safe use.

    Args:
        text: Text to sanitize.
        max_length: Maximum allowed length.

    Returns:
        Sanitized string.
    """
    if not isinstance(text, str):
        return ""

    # Strip whitespace
    text = text.strip()

    # Truncate if too long
    if len(text) > max_length:
        text = text[:max_length]

    # Escape HTML
    text = html.escape(text)

    return text


def validate_model_name(model: str) -> bool:
    """
    Validate a model name format.

    Args:
        model: Model name to validate.

    Returns:
        True if valid.
    """
    if not model or not isinstance(model, str):
        return False

    # Allow alphanumeric, hyphens, underscores, dots, and forward slashes
    pattern = r"^[a-zA-Z0-9][a-zA-Z0-9\-_./]*$"
    return bool(re.match(pattern, model))


def check_prompt_injection(text: str) -> tuple[bool, list[str]]:
    """
    Check text for potential prompt injection.

    Args:
        text: Text to check.

    Returns:
        Tuple of (has_injection, matched_patterns).
    """
    validator = ContentValidator()
    matches = []

    for pattern in validator._injection_regex:
        match = pattern.search(text)
        if match:
            matches.append(match.group())

    return len(matches) > 0, matches
