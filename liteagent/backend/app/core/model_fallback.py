"""
Model fallback chain for automatic failover between LLM models.
Provides resilient model selection with configurable fallback strategies.
"""
import asyncio
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, TypeVar

T = TypeVar("T")


class FallbackReason(str, Enum):
    """Reasons for triggering fallback."""

    ERROR = "error"  # Model returned an error
    TIMEOUT = "timeout"  # Request timed out
    RATE_LIMIT = "rate_limit"  # Hit rate limits
    CAPACITY = "capacity"  # Model at capacity
    UNAVAILABLE = "unavailable"  # Model not available
    CIRCUIT_OPEN = "circuit_open"  # Circuit breaker is open


class SelectionStrategy(str, Enum):
    """Model selection strategies."""

    PRIORITY = "priority"  # Use models in priority order
    ROUND_ROBIN = "round_robin"  # Rotate between models
    RANDOM = "random"  # Random selection
    WEIGHTED = "weighted"  # Weighted random selection
    LEAST_ERRORS = "least_errors"  # Prefer models with fewer errors


@dataclass
class ModelConfig:
    """Configuration for a model in the fallback chain."""

    model_id: str
    provider: str
    priority: int = 0  # Lower = higher priority
    weight: float = 1.0  # For weighted selection
    timeout_seconds: float = 30.0
    max_retries: int = 2
    enabled: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelStatus:
    """Runtime status of a model."""

    model_id: str
    is_healthy: bool = True
    error_count: int = 0
    success_count: int = 0
    last_error: str | None = None
    last_error_time: datetime | None = None
    last_success_time: datetime | None = None
    avg_latency_ms: float = 0.0

    def record_success(self, latency_ms: float) -> None:
        """Record a successful request."""
        self.success_count += 1
        self.last_success_time = datetime.now(timezone.utc)
        self.is_healthy = True

        # Update moving average latency
        if self.avg_latency_ms == 0:
            self.avg_latency_ms = latency_ms
        else:
            # Exponential moving average
            self.avg_latency_ms = 0.9 * self.avg_latency_ms + 0.1 * latency_ms

    def record_error(self, error: str) -> None:
        """Record an error."""
        self.error_count += 1
        self.last_error = error
        self.last_error_time = datetime.now(timezone.utc)

        # Mark unhealthy after consecutive errors
        if self.error_count > 3:
            self.is_healthy = False

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        total = self.success_count + self.error_count
        if total == 0:
            return 1.0
        return self.success_count / total


@dataclass
class FallbackAttempt:
    """Record of a fallback attempt."""

    model_id: str
    success: bool
    latency_ms: float
    error: str | None = None
    reason: FallbackReason | None = None


@dataclass
class FallbackResult:
    """Result of a fallback chain execution."""

    result: Any
    model_used: str
    attempts: list[FallbackAttempt] = field(default_factory=list)
    total_latency_ms: float = 0.0

    @property
    def fallback_count(self) -> int:
        """Number of fallback attempts (excluding final success)."""
        return len(self.attempts) - 1 if self.attempts else 0


class ModelSelector(ABC):
    """Base class for model selection strategies."""

    @abstractmethod
    def select(
        self,
        models: list[ModelConfig],
        statuses: dict[str, ModelStatus],
    ) -> list[ModelConfig]:
        """
        Select models in order of preference.

        Args:
            models: Available model configurations.
            statuses: Current model statuses.

        Returns:
            Ordered list of models to try.
        """
        pass


class PrioritySelector(ModelSelector):
    """Select models by priority order."""

    def select(
        self,
        models: list[ModelConfig],
        statuses: dict[str, ModelStatus],
    ) -> list[ModelConfig]:
        """Select by priority (lower = higher priority)."""
        enabled = [m for m in models if m.enabled]
        return sorted(enabled, key=lambda m: m.priority)


class RoundRobinSelector(ModelSelector):
    """Rotate between models."""

    def __init__(self):
        self._index = 0

    def select(
        self,
        models: list[ModelConfig],
        statuses: dict[str, ModelStatus],
    ) -> list[ModelConfig]:
        """Select using round-robin rotation."""
        enabled = [m for m in models if m.enabled]
        if not enabled:
            return []

        # Rotate starting point
        self._index = (self._index + 1) % len(enabled)

        # Return rotated list
        return enabled[self._index :] + enabled[: self._index]


class WeightedSelector(ModelSelector):
    """Random weighted selection."""

    def select(
        self,
        models: list[ModelConfig],
        statuses: dict[str, ModelStatus],
    ) -> list[ModelConfig]:
        """Select with weighted random."""
        enabled = [m for m in models if m.enabled]
        if not enabled:
            return []

        # Calculate weights considering health
        weights = []
        for model in enabled:
            status = statuses.get(model.model_id)
            weight = model.weight
            if status and not status.is_healthy:
                weight *= 0.1  # Reduce weight for unhealthy models
            weights.append(weight)

        # Sort by weighted random
        paired = list(zip(enabled, weights))
        random.shuffle(paired)
        paired.sort(key=lambda x: random.random() * x[1], reverse=True)

        return [p[0] for p in paired]


class LeastErrorsSelector(ModelSelector):
    """Select models with fewest errors."""

    def select(
        self,
        models: list[ModelConfig],
        statuses: dict[str, ModelStatus],
    ) -> list[ModelConfig]:
        """Select by error count (fewer = higher priority)."""
        enabled = [m for m in models if m.enabled]

        def error_score(model: ModelConfig) -> tuple[int, float]:
            status = statuses.get(model.model_id)
            if not status:
                return (0, model.priority)
            return (status.error_count, model.priority)

        return sorted(enabled, key=error_score)


class ModelFallbackChain:
    """
    Manages fallback between multiple models.
    Automatically tries backup models when primary fails.
    """

    def __init__(
        self,
        models: list[ModelConfig],
        strategy: SelectionStrategy = SelectionStrategy.PRIORITY,
    ):
        """
        Initialize fallback chain.

        Args:
            models: List of model configurations.
            strategy: Model selection strategy.
        """
        self.models = models
        self._statuses: dict[str, ModelStatus] = {
            m.model_id: ModelStatus(model_id=m.model_id) for m in models
        }
        self._selector = self._create_selector(strategy)

    def _create_selector(self, strategy: SelectionStrategy) -> ModelSelector:
        """Create selector based on strategy."""
        selectors = {
            SelectionStrategy.PRIORITY: PrioritySelector,
            SelectionStrategy.ROUND_ROBIN: RoundRobinSelector,
            SelectionStrategy.WEIGHTED: WeightedSelector,
            SelectionStrategy.LEAST_ERRORS: LeastErrorsSelector,
            SelectionStrategy.RANDOM: WeightedSelector,  # Random uses weighted with equal weights
        }
        return selectors[strategy]()

    def get_ordered_models(self) -> list[ModelConfig]:
        """Get models in order of preference."""
        return self._selector.select(self.models, self._statuses)

    def get_status(self, model_id: str) -> ModelStatus | None:
        """Get status for a model."""
        return self._statuses.get(model_id)

    def get_all_statuses(self) -> dict[str, ModelStatus]:
        """Get all model statuses."""
        return self._statuses.copy()

    async def execute(
        self,
        func: Callable[..., T],
        *args,
        **kwargs,
    ) -> FallbackResult:
        """
        Execute a function with fallback support.

        The function should accept 'model' as keyword argument.

        Args:
            func: Async function to execute.
            *args: Function arguments.
            **kwargs: Function keyword arguments.

        Returns:
            FallbackResult with result and attempt history.

        Raises:
            Exception: If all models fail.
        """
        attempts: list[FallbackAttempt] = []
        total_start = time.monotonic()
        ordered_models = self.get_ordered_models()

        if not ordered_models:
            raise ValueError("No models available in fallback chain")

        last_error = None

        for model_config in ordered_models:
            model_id = model_config.model_id
            start = time.monotonic()

            try:
                # Execute with this model
                result = await asyncio.wait_for(
                    func(*args, model=model_id, **kwargs),
                    timeout=model_config.timeout_seconds,
                )

                latency = (time.monotonic() - start) * 1000
                self._statuses[model_id].record_success(latency)

                attempts.append(
                    FallbackAttempt(
                        model_id=model_id,
                        success=True,
                        latency_ms=latency,
                    )
                )

                return FallbackResult(
                    result=result,
                    model_used=model_id,
                    attempts=attempts,
                    total_latency_ms=(time.monotonic() - total_start) * 1000,
                )

            except asyncio.TimeoutError:
                latency = (time.monotonic() - start) * 1000
                error_msg = "Request timed out"
                self._statuses[model_id].record_error(error_msg)
                last_error = TimeoutError(error_msg)

                attempts.append(
                    FallbackAttempt(
                        model_id=model_id,
                        success=False,
                        latency_ms=latency,
                        error=error_msg,
                        reason=FallbackReason.TIMEOUT,
                    )
                )

            except Exception as e:
                latency = (time.monotonic() - start) * 1000
                error_msg = str(e)
                self._statuses[model_id].record_error(error_msg)
                last_error = e

                reason = self._classify_error(e)
                attempts.append(
                    FallbackAttempt(
                        model_id=model_id,
                        success=False,
                        latency_ms=latency,
                        error=error_msg,
                        reason=reason,
                    )
                )

        # All models failed
        raise last_error or Exception("All models in fallback chain failed")

    def _classify_error(self, error: Exception) -> FallbackReason:
        """Classify error for reporting."""
        error_str = str(error).lower()

        if "rate limit" in error_str:
            return FallbackReason.RATE_LIMIT
        elif "timeout" in error_str:
            return FallbackReason.TIMEOUT
        elif "capacity" in error_str or "overloaded" in error_str:
            return FallbackReason.CAPACITY
        elif "unavailable" in error_str or "503" in error_str:
            return FallbackReason.UNAVAILABLE
        else:
            return FallbackReason.ERROR

    def add_model(self, model: ModelConfig) -> None:
        """Add a model to the chain."""
        self.models.append(model)
        self._statuses[model.model_id] = ModelStatus(model_id=model.model_id)

    def remove_model(self, model_id: str) -> None:
        """Remove a model from the chain."""
        self.models = [m for m in self.models if m.model_id != model_id]
        self._statuses.pop(model_id, None)

    def enable_model(self, model_id: str) -> None:
        """Enable a model."""
        for model in self.models:
            if model.model_id == model_id:
                model.enabled = True
                break

    def disable_model(self, model_id: str) -> None:
        """Disable a model."""
        for model in self.models:
            if model.model_id == model_id:
                model.enabled = False
                break

    def reset_status(self, model_id: str | None = None) -> None:
        """Reset status for one or all models."""
        if model_id:
            if model_id in self._statuses:
                self._statuses[model_id] = ModelStatus(model_id=model_id)
        else:
            self._statuses = {
                m.model_id: ModelStatus(model_id=m.model_id) for m in self.models
            }


# Factory functions
def create_fallback_chain(
    model_ids: list[str],
    provider: str = "openai",
    strategy: SelectionStrategy = SelectionStrategy.PRIORITY,
) -> ModelFallbackChain:
    """
    Create a simple fallback chain.

    Args:
        model_ids: List of model IDs in priority order.
        provider: Provider name.
        strategy: Selection strategy.

    Returns:
        Configured fallback chain.
    """
    models = [
        ModelConfig(
            model_id=model_id,
            provider=provider,
            priority=i,
        )
        for i, model_id in enumerate(model_ids)
    ]
    return ModelFallbackChain(models, strategy)


def create_multi_provider_chain(
    models: list[dict[str, Any]],
    strategy: SelectionStrategy = SelectionStrategy.PRIORITY,
) -> ModelFallbackChain:
    """
    Create a multi-provider fallback chain.

    Args:
        models: List of model dicts with 'model_id', 'provider', etc.
        strategy: Selection strategy.

    Returns:
        Configured fallback chain.
    """
    configs = [ModelConfig(**m) for m in models]
    return ModelFallbackChain(configs, strategy)
