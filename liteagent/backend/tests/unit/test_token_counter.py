"""
Unit tests for token counting.
Tests token estimation for different models.
"""
import pytest
from unittest.mock import Mock, patch

from app.core.token_counter import (
    TokenCounter,
    count_tokens,
    count_message_tokens,
    estimate_cost,
    ModelPricing,
    get_model_pricing,
)


class TestTokenCounter:
    """Tests for token counter."""

    def test_count_simple_text(self):
        """Test counting tokens in simple text."""
        counter = TokenCounter(model="gpt-4o")
        count = counter.count("Hello, world!")
        assert count > 0
        assert count < 10  # Simple text should be few tokens

    def test_count_longer_text(self):
        """Test counting tokens in longer text."""
        counter = TokenCounter(model="gpt-4o")
        short_count = counter.count("Hi")
        long_count = counter.count("This is a much longer sentence with more words.")
        assert long_count > short_count

    def test_count_empty_string(self):
        """Test counting tokens in empty string."""
        counter = TokenCounter(model="gpt-4o")
        count = counter.count("")
        assert count == 0

    def test_different_models_same_tokenizer(self):
        """Test GPT-4 and GPT-3.5 use same tokenizer family."""
        counter_4 = TokenCounter(model="gpt-4o")
        counter_35 = TokenCounter(model="gpt-3.5-turbo")

        text = "This is a test sentence."
        # Both should give similar results (same tokenizer)
        count_4 = counter_4.count(text)
        count_35 = counter_35.count(text)
        assert abs(count_4 - count_35) < 2


class TestCountTokens:
    """Tests for count_tokens function."""

    def test_count_tokens_default_model(self):
        """Test counting with default model."""
        count = count_tokens("Hello, world!")
        assert count > 0

    def test_count_tokens_specific_model(self):
        """Test counting for specific model."""
        count = count_tokens("Hello, world!", model="gpt-4o")
        assert count > 0


class TestCountMessageTokens:
    """Tests for message token counting."""

    def test_count_single_message(self):
        """Test counting tokens in a single message."""
        messages = [{"role": "user", "content": "Hello!"}]
        count = count_message_tokens(messages, model="gpt-4o")
        assert count > 0

    def test_count_multiple_messages(self):
        """Test counting tokens in multiple messages."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there! How can I help?"},
        ]
        count = count_message_tokens(messages, model="gpt-4o")
        assert count > 10  # Should be more than a simple message

    def test_includes_message_overhead(self):
        """Test token count includes message formatting overhead."""
        # A message has overhead for role, content markers, etc.
        content = "Hi"
        content_tokens = count_tokens(content, model="gpt-4o")
        message_tokens = count_message_tokens(
            [{"role": "user", "content": content}],
            model="gpt-4o",
        )
        assert message_tokens > content_tokens

    def test_empty_messages(self):
        """Test counting empty message list."""
        count = count_message_tokens([], model="gpt-4o")
        assert count >= 0


class TestModelPricing:
    """Tests for model pricing."""

    def test_pricing_structure(self):
        """Test pricing has required fields."""
        pricing = ModelPricing(
            input_price_per_1k=0.03,
            output_price_per_1k=0.06,
        )
        assert pricing.input_price_per_1k == 0.03
        assert pricing.output_price_per_1k == 0.06

    def test_get_known_model_pricing(self):
        """Test getting pricing for known models."""
        pricing = get_model_pricing("gpt-4o")
        assert pricing is not None
        assert pricing.input_price_per_1k > 0

    def test_get_unknown_model_pricing(self):
        """Test getting pricing for unknown model returns None."""
        pricing = get_model_pricing("unknown-model-xyz")
        assert pricing is None


class TestEstimateCost:
    """Tests for cost estimation."""

    def test_estimate_input_cost(self):
        """Test estimating input token cost."""
        cost = estimate_cost(
            model="gpt-4o",
            input_tokens=1000,
            output_tokens=0,
        )
        assert cost > 0

    def test_estimate_output_cost(self):
        """Test estimating output token cost."""
        cost = estimate_cost(
            model="gpt-4o",
            input_tokens=0,
            output_tokens=1000,
        )
        assert cost > 0

    def test_output_more_expensive_than_input(self):
        """Test output tokens cost more than input for most models."""
        input_cost = estimate_cost(model="gpt-4o", input_tokens=1000, output_tokens=0)
        output_cost = estimate_cost(model="gpt-4o", input_tokens=0, output_tokens=1000)
        assert output_cost >= input_cost  # Output is typically more expensive

    def test_estimate_combined_cost(self):
        """Test estimating combined input and output cost."""
        combined = estimate_cost(model="gpt-4o", input_tokens=500, output_tokens=500)
        input_only = estimate_cost(model="gpt-4o", input_tokens=500, output_tokens=0)
        output_only = estimate_cost(model="gpt-4o", input_tokens=0, output_tokens=500)
        assert abs(combined - (input_only + output_only)) < 0.0001

    def test_unknown_model_returns_zero(self):
        """Test unknown model returns zero cost."""
        cost = estimate_cost(model="unknown-model", input_tokens=1000, output_tokens=1000)
        assert cost == 0
