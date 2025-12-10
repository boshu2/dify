"""
Output validator for sandbox execution results.
"""

from dataclasses import dataclass, field
from typing import Any

from app.config import SandboxConfig, get_config


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
    warnings: list[str] = field(default_factory=list)


class OutputValidator:
    """Validates execution output against configured limits."""

    def __init__(self, config: SandboxConfig | None = None):
        self.config = config or get_config()

    def validate(self, output: Any) -> ValidationResult:
        """Validate execution output."""
        warnings: list[str] = []

        try:
            validated = self._validate_value(output, depth=0, warnings=warnings)
            return ValidationResult(
                valid=True,
                value=validated,
                warnings=warnings,
            )
        except ValidationError as e:
            return ValidationResult(
                valid=False,
                error=str(e),
            )

    def _validate_value(self, value: Any, depth: int, warnings: list[str]) -> Any:
        """Recursively validate a value."""
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
        return str(value)[: self.config.max_string_length]

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

        # Round to max precision
        return round(value, self.config.max_precision)

    def _validate_string(self, value: str, warnings: list[str]) -> str:
        """Validate string value."""
        # Strip null characters
        value = value.replace("\x00", "")

        if len(value) > self.config.max_string_length:
            warnings.append(
                f"String truncated from {len(value)} to {self.config.max_string_length} chars"
            )
            return value[: self.config.max_string_length]
        return value

    def _validate_array(
        self, value: list | tuple, depth: int, warnings: list[str]
    ) -> list:
        """Validate array value."""
        if len(value) > self.config.max_array_length:
            warnings.append(
                f"Array truncated from {len(value)} to {self.config.max_array_length} items"
            )
            value = value[: self.config.max_array_length]

        return [self._validate_value(item, depth + 1, warnings) for item in value]

    def _validate_dict(self, value: dict, depth: int, warnings: list[str]) -> dict:
        """Validate dictionary value."""
        return {
            self._validate_key(k): self._validate_value(v, depth + 1, warnings)
            for k, v in value.items()
        }

    def _validate_key(self, key: Any) -> str:
        """Validate dictionary key."""
        if not isinstance(key, str):
            key = str(key)
        return key[:1000]


def sanitize_for_json(value: Any, max_depth: int = 5) -> Any:
    """Sanitize value for JSON serialization."""
    if max_depth <= 0:
        return str(value)

    if value is None:
        return None

    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)):
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
        return {str(k): sanitize_for_json(v, max_depth - 1) for k, v in value.items()}

    return str(value)
