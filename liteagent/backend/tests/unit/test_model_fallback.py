"""
Unit tests for model fallback chain.
Tests automatic failover between models.
"""
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock
from datetime import datetime, timezone

from app.core.model_fallback import (
    FallbackReason,
    SelectionStrategy,
    ModelConfig,
    ModelStatus,
    FallbackAttempt,
    FallbackResult,
    PrioritySelector,
    RoundRobinSelector,
    WeightedSelector,
    LeastErrorsSelector,
    ModelFallbackChain,
    create_fallback_chain,
    create_multi_provider_chain,
)


class TestFallbackReason:
    """Tests for fallback reason enum."""

    def test_reason_values(self):
        """Test reason enum values."""
        assert FallbackReason.ERROR.value == "error"
        assert FallbackReason.TIMEOUT.value == "timeout"
        assert FallbackReason.RATE_LIMIT.value == "rate_limit"


class TestSelectionStrategy:
    """Tests for selection strategy enum."""

    def test_strategy_values(self):
        """Test strategy enum values."""
        assert SelectionStrategy.PRIORITY.value == "priority"
        assert SelectionStrategy.ROUND_ROBIN.value == "round_robin"
        assert SelectionStrategy.WEIGHTED.value == "weighted"


class TestModelConfig:
    """Tests for model configuration."""

    def test_create_basic_config(self):
        """Test creating basic config."""
        config = ModelConfig(
            model_id="gpt-4",
            provider="openai",
        )

        assert config.model_id == "gpt-4"
        assert config.provider == "openai"
        assert config.enabled is True

    def test_config_with_priority(self):
        """Test config with priority."""
        config = ModelConfig(
            model_id="gpt-4",
            provider="openai",
            priority=1,
        )

        assert config.priority == 1

    def test_config_with_timeout(self):
        """Test config with timeout."""
        config = ModelConfig(
            model_id="gpt-4",
            provider="openai",
            timeout_seconds=60.0,
        )

        assert config.timeout_seconds == 60.0


class TestModelStatus:
    """Tests for model status tracking."""

    def test_initial_status(self):
        """Test initial status values."""
        status = ModelStatus(model_id="gpt-4")

        assert status.is_healthy is True
        assert status.error_count == 0
        assert status.success_count == 0

    def test_record_success(self):
        """Test recording a success."""
        status = ModelStatus(model_id="gpt-4")
        status.record_success(latency_ms=100.0)

        assert status.success_count == 1
        assert status.is_healthy is True
        assert status.last_success_time is not None

    def test_record_error(self):
        """Test recording an error."""
        status = ModelStatus(model_id="gpt-4")
        status.record_error("Connection failed")

        assert status.error_count == 1
        assert status.last_error == "Connection failed"

    def test_mark_unhealthy_after_errors(self):
        """Test model marked unhealthy after multiple errors."""
        status = ModelStatus(model_id="gpt-4")

        for _ in range(5):
            status.record_error("Error")

        assert status.is_healthy is False

    def test_success_rate(self):
        """Test success rate calculation."""
        status = ModelStatus(model_id="gpt-4")
        status.success_count = 8
        status.error_count = 2

        assert status.success_rate == 0.8

    def test_success_rate_no_requests(self):
        """Test success rate with no requests."""
        status = ModelStatus(model_id="gpt-4")

        assert status.success_rate == 1.0

    def test_latency_tracking(self):
        """Test average latency tracking."""
        status = ModelStatus(model_id="gpt-4")

        status.record_success(100.0)
        status.record_success(200.0)

        # Should be exponential moving average
        assert status.avg_latency_ms > 0


class TestFallbackResult:
    """Tests for fallback result."""

    def test_create_result(self):
        """Test creating a result."""
        result = FallbackResult(
            result="success",
            model_used="gpt-4",
            attempts=[
                FallbackAttempt(model_id="gpt-4", success=True, latency_ms=100),
            ],
        )

        assert result.result == "success"
        assert result.model_used == "gpt-4"

    def test_fallback_count(self):
        """Test fallback count calculation."""
        result = FallbackResult(
            result="success",
            model_used="gpt-3.5",
            attempts=[
                FallbackAttempt(model_id="gpt-4", success=False, latency_ms=50),
                FallbackAttempt(model_id="gpt-3.5", success=True, latency_ms=100),
            ],
        )

        assert result.fallback_count == 1


class TestPrioritySelector:
    """Tests for priority-based selection."""

    def test_select_by_priority(self):
        """Test selection by priority order."""
        selector = PrioritySelector()
        models = [
            ModelConfig(model_id="model-c", provider="p", priority=2),
            ModelConfig(model_id="model-a", provider="p", priority=0),
            ModelConfig(model_id="model-b", provider="p", priority=1),
        ]

        result = selector.select(models, {})

        assert result[0].model_id == "model-a"
        assert result[1].model_id == "model-b"
        assert result[2].model_id == "model-c"

    def test_excludes_disabled(self):
        """Test disabled models are excluded."""
        selector = PrioritySelector()
        models = [
            ModelConfig(model_id="model-a", provider="p", priority=0, enabled=False),
            ModelConfig(model_id="model-b", provider="p", priority=1),
        ]

        result = selector.select(models, {})

        assert len(result) == 1
        assert result[0].model_id == "model-b"


class TestRoundRobinSelector:
    """Tests for round-robin selection."""

    def test_rotates_selection(self):
        """Test selection rotates."""
        selector = RoundRobinSelector()
        models = [
            ModelConfig(model_id="a", provider="p"),
            ModelConfig(model_id="b", provider="p"),
            ModelConfig(model_id="c", provider="p"),
        ]

        # Call multiple times to see rotation
        first = selector.select(models, {})
        second = selector.select(models, {})
        third = selector.select(models, {})

        # Each call should start from different model
        assert first[0].model_id != second[0].model_id


class TestWeightedSelector:
    """Tests for weighted selection."""

    def test_respects_weights(self):
        """Test weighted selection (statistical)."""
        selector = WeightedSelector()
        models = [
            ModelConfig(model_id="heavy", provider="p", weight=10.0),
            ModelConfig(model_id="light", provider="p", weight=0.1),
        ]

        # Run multiple selections
        heavy_first_count = 0
        for _ in range(100):
            result = selector.select(models, {})
            if result[0].model_id == "heavy":
                heavy_first_count += 1

        # Heavy should be selected first more often
        assert heavy_first_count > 50


class TestLeastErrorsSelector:
    """Tests for least errors selection."""

    def test_prefers_fewer_errors(self):
        """Test selection prefers models with fewer errors."""
        selector = LeastErrorsSelector()
        models = [
            ModelConfig(model_id="model-a", provider="p"),
            ModelConfig(model_id="model-b", provider="p"),
        ]
        statuses = {
            "model-a": ModelStatus(model_id="model-a"),
            "model-b": ModelStatus(model_id="model-b"),
        }
        statuses["model-a"].error_count = 10
        statuses["model-b"].error_count = 0

        result = selector.select(models, statuses)

        assert result[0].model_id == "model-b"


class TestModelFallbackChain:
    """Tests for model fallback chain."""

    def test_create_chain(self):
        """Test creating a fallback chain."""
        models = [
            ModelConfig(model_id="gpt-4", provider="openai"),
            ModelConfig(model_id="gpt-3.5", provider="openai"),
        ]
        chain = ModelFallbackChain(models)

        assert len(chain.models) == 2

    def test_get_ordered_models(self):
        """Test getting ordered models."""
        models = [
            ModelConfig(model_id="model-b", provider="p", priority=1),
            ModelConfig(model_id="model-a", provider="p", priority=0),
        ]
        chain = ModelFallbackChain(models)

        ordered = chain.get_ordered_models()

        assert ordered[0].model_id == "model-a"

    @pytest.mark.asyncio
    async def test_execute_success(self):
        """Test successful execution."""
        models = [
            ModelConfig(model_id="gpt-4", provider="openai"),
        ]
        chain = ModelFallbackChain(models)

        async def mock_func(model: str):
            return f"Response from {model}"

        result = await chain.execute(mock_func)

        assert result.result == "Response from gpt-4"
        assert result.model_used == "gpt-4"
        assert result.fallback_count == 0

    @pytest.mark.asyncio
    async def test_execute_fallback(self):
        """Test fallback to second model."""
        models = [
            ModelConfig(model_id="gpt-4", provider="openai"),
            ModelConfig(model_id="gpt-3.5", provider="openai"),
        ]
        chain = ModelFallbackChain(models)

        call_count = 0

        async def mock_func(model: str):
            nonlocal call_count
            call_count += 1
            if model == "gpt-4":
                raise Exception("Model unavailable")
            return f"Response from {model}"

        result = await chain.execute(mock_func)

        assert result.model_used == "gpt-3.5"
        assert result.fallback_count == 1
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_execute_timeout(self):
        """Test timeout handling."""
        models = [
            ModelConfig(model_id="slow", provider="p", timeout_seconds=0.1),
            ModelConfig(model_id="fast", provider="p"),
        ]
        chain = ModelFallbackChain(models)

        async def mock_func(model: str):
            if model == "slow":
                await asyncio.sleep(1)  # Will timeout
            return "fast response"

        result = await chain.execute(mock_func)

        assert result.model_used == "fast"
        assert any(a.reason == FallbackReason.TIMEOUT for a in result.attempts)

    @pytest.mark.asyncio
    async def test_execute_all_fail(self):
        """Test all models failing."""
        models = [
            ModelConfig(model_id="model-a", provider="p"),
            ModelConfig(model_id="model-b", provider="p"),
        ]
        chain = ModelFallbackChain(models)

        async def mock_func(model: str):
            raise Exception("Always fails")

        with pytest.raises(Exception) as exc:
            await chain.execute(mock_func)

        assert "Always fails" in str(exc.value)

    def test_add_model(self):
        """Test adding a model."""
        chain = ModelFallbackChain([])
        chain.add_model(ModelConfig(model_id="new", provider="p"))

        assert len(chain.models) == 1

    def test_remove_model(self):
        """Test removing a model."""
        models = [
            ModelConfig(model_id="keep", provider="p"),
            ModelConfig(model_id="remove", provider="p"),
        ]
        chain = ModelFallbackChain(models)
        chain.remove_model("remove")

        assert len(chain.models) == 1
        assert chain.models[0].model_id == "keep"

    def test_enable_disable_model(self):
        """Test enabling/disabling models."""
        models = [
            ModelConfig(model_id="test", provider="p"),
        ]
        chain = ModelFallbackChain(models)

        chain.disable_model("test")
        assert chain.models[0].enabled is False

        chain.enable_model("test")
        assert chain.models[0].enabled is True

    def test_get_status(self):
        """Test getting model status."""
        models = [ModelConfig(model_id="test", provider="p")]
        chain = ModelFallbackChain(models)

        status = chain.get_status("test")

        assert status is not None
        assert status.model_id == "test"

    def test_reset_status(self):
        """Test resetting model status."""
        models = [ModelConfig(model_id="test", provider="p")]
        chain = ModelFallbackChain(models)

        # Record some activity
        chain._statuses["test"].record_error("Error")
        chain.reset_status("test")

        assert chain._statuses["test"].error_count == 0


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_fallback_chain(self):
        """Test creating a simple fallback chain."""
        chain = create_fallback_chain(
            model_ids=["gpt-4", "gpt-3.5-turbo"],
            provider="openai",
        )

        assert len(chain.models) == 2
        assert chain.models[0].model_id == "gpt-4"

    def test_create_multi_provider_chain(self):
        """Test creating multi-provider chain."""
        chain = create_multi_provider_chain([
            {"model_id": "gpt-4", "provider": "openai", "priority": 0},
            {"model_id": "claude-3", "provider": "anthropic", "priority": 1},
        ])

        assert len(chain.models) == 2
        assert chain.models[0].provider == "openai"
        assert chain.models[1].provider == "anthropic"


class TestErrorClassification:
    """Tests for error classification."""

    def test_classify_rate_limit(self):
        """Test rate limit error classification."""
        models = [ModelConfig(model_id="test", provider="p")]
        chain = ModelFallbackChain(models)

        reason = chain._classify_error(Exception("rate limit exceeded"))

        assert reason == FallbackReason.RATE_LIMIT

    def test_classify_timeout(self):
        """Test timeout error classification."""
        models = [ModelConfig(model_id="test", provider="p")]
        chain = ModelFallbackChain(models)

        reason = chain._classify_error(Exception("Request timeout"))

        assert reason == FallbackReason.TIMEOUT

    def test_classify_unavailable(self):
        """Test unavailable error classification."""
        models = [ModelConfig(model_id="test", provider="p")]
        chain = ModelFallbackChain(models)

        reason = chain._classify_error(Exception("Service unavailable"))

        assert reason == FallbackReason.UNAVAILABLE

    def test_classify_generic(self):
        """Test generic error classification."""
        models = [ModelConfig(model_id="test", provider="p")]
        chain = ModelFallbackChain(models)

        reason = chain._classify_error(Exception("Some random error"))

        assert reason == FallbackReason.ERROR
