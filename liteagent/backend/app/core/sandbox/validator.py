"""
Output validator for sandbox execution results.

Validates and sanitizes execution output to prevent:
- Resource exhaustion
- Data exfiltration
- Injection attacks
"""

from dataclasses import dataclass
from typing import Any

from app.core.sandbox.config import SandboxConfig, get_sandbox_config


class ValidationError(Exception):
    """Error during output validation."""

    def __init__(self, message: str, field: str | None = None):
        super().__init__(message)
        self.field = field


@dataclass
class ValidationResult:
    """Result of output validation."""

    valid: bool
    value: Any = None
    error: str | None = None
    warnings: list[str] | None = None


class OutputValidator:
    """
    Validates execution output against configured limits.

    Checks:
    - String length
    - Number range
    - Array length
    - Object depth
    - Data types
    """

    def __init__(self, config: SandboxConfig | None = None):
        self.config = config or get_sandbox_config()

    def validate(self, output: Any) -> ValidationResult:
        """
        Validate execution output.

        Args:
            output: Output to validate

        Returns:
            ValidationResult with validated value or error
        """
        warnings: list[str] = []

        try:
            validated = self._validate_value(output, depth=0, warnings=warnings)
            return ValidationResult(
                valid=True,
                value=validated,
                warnings=warnings if warnings else None,
            )
        except ValidationError as e:
            return ValidationResult(
                valid=False,
                error=str(e),
            )

    def _validate_value(
        self,
        value: Any,
        depth: int,
        warnings: list[str],
    ) -> Any:
        """Recursively validate a value."""
        # Check depth
        if depth > self.config.max_depth:
            raise ValidationError(
                f"Object depth exceeds maximum of {self.config.max_depth}",
                field="depth",
            )

        if value is None:
            return None

        if isinstance(value, bool):
            return value

        if isinstance(value, int):
            return self._validate_int(value)

        if isinstance(value, float):
            return self._validate_float(value)

        if isinstance(value, str):
            return self._validate_string(value, warnings)

        if isinstance(value, (list, tuple)):
            return self._validate_array(value, depth, warnings)

        if isinstance(value, dict):
            return self._validate_dict(value, depth, warnings)

        # Unknown type - convert to string
        warnings.append(f"Unknown type {type(value).__name__} converted to string")
        return str(value)[:self.config.max_string_length]

    def _validate_int(self, value: int) -> int:
        """Validate integer value."""
        if value > self.config.max_number:
            raise ValidationError(
                f"Number {value} exceeds maximum of {self.config.max_number}",
                field="number",
            )
        if value < self.config.min_number:
            raise ValidationError(
                f"Number {value} below minimum of {self.config.min_number}",
                field="number",
            )
        return value

    def _validate_float(self, value: float) -> float:
        """Validate float value."""
        if value > self.config.max_number:
            raise ValidationError(
                f"Number {value} exceeds maximum of {self.config.max_number}",
                field="number",
            )
        if value < self.config.min_number:
            raise ValidationError(
                f"Number {value} below minimum of {self.config.min_number}",
                field="number",
            )

        # Check precision
        str_value = str(value)
        if "." in str_value:
            decimal_places = len(str_value.split(".")[1])
            if decimal_places > self.config.max_precision:
                # Round to max precision
                return round(value, self.config.max_precision)

        return value

    def _validate_string(self, value: str, warnings: list[str]) -> str:
        """Validate string value."""
        if len(value) > self.config.max_string_length:
            warnings.append(
                f"String truncated from {len(value)} to {self.config.max_string_length} chars"
            )
            return value[:self.config.max_string_length]
        return value

    def _validate_array(
        self,
        value: list | tuple,
        depth: int,
        warnings: list[str],
    ) -> list:
        """Validate array value."""
        if len(value) > self.config.max_array_length:
            warnings.append(
                f"Array truncated from {len(value)} to {self.config.max_array_length} items"
            )
            value = value[:self.config.max_array_length]

        return [
            self._validate_value(item, depth + 1, warnings)
            for item in value
        ]

    def _validate_dict(
        self,
        value: dict,
        depth: int,
        warnings: list[str],
    ) -> dict:
        """Validate dictionary value."""
        return {
            self._validate_key(k): self._validate_value(v, depth + 1, warnings)
            for k, v in value.items()
        }

    def _validate_key(self, key: Any) -> str:
        """Validate dictionary key."""
        if not isinstance(key, str):
            key = str(key)
        return key[:1000]  # Limit key length

    def validate_and_sanitize(self, output: Any) -> Any:
        """
        Validate and sanitize output, raising on errors.

        Args:
            output: Output to validate

        Returns:
            Sanitized output

        Raises:
            ValidationError: If validation fails
        """
        result = self.validate(output)

        if not result.valid:
            raise ValidationError(result.error or "Validation failed")

        return result.value


# Type checking functions
def is_safe_type(value: Any) -> bool:
    """Check if value is a safe type."""
    if value is None:
        return True
    if isinstance(value, (bool, int, float, str)):
        return True
    if isinstance(value, (list, tuple)):
        return all(is_safe_type(item) for item in value)
    if isinstance(value, dict):
        return all(
            isinstance(k, str) and is_safe_type(v)
            for k, v in value.items()
        )
    return False


def sanitize_for_json(value: Any, max_depth: int = 5) -> Any:
    """
    Sanitize value for JSON serialization.

    Args:
        value: Value to sanitize
        max_depth: Maximum recursion depth

    Returns:
        JSON-safe value
    """
    if max_depth <= 0:
        return str(value)

    if value is None:
        return None

    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)):
        # Handle special float values
        if isinstance(value, float):
            if value != value:  # NaN
                return None
            if value == float("inf") or value == float("-inf"):
                return None
        return value

    if isinstance(value, str):
        return value

    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return value.hex()

    if isinstance(value, (list, tuple)):
        return [sanitize_for_json(item, max_depth - 1) for item in value]

    if isinstance(value, dict):
        return {
            str(k): sanitize_for_json(v, max_depth - 1)
            for k, v in value.items()
        }

    # Convert other types to string
    return str(value)
