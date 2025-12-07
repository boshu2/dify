"""
Token counting and cost estimation for LLM requests.
Uses tiktoken for accurate token counting.
"""
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import tiktoken


@dataclass
class ModelPricing:
    """Pricing for a model (per 1K tokens)."""

    input_price_per_1k: float
    output_price_per_1k: float


# Model pricing (as of early 2024, prices in USD)
MODEL_PRICING: dict[str, ModelPricing] = {
    # GPT-4o
    "gpt-4o": ModelPricing(input_price_per_1k=0.005, output_price_per_1k=0.015),
    "gpt-4o-mini": ModelPricing(input_price_per_1k=0.00015, output_price_per_1k=0.0006),
    # GPT-4
    "gpt-4-turbo": ModelPricing(input_price_per_1k=0.01, output_price_per_1k=0.03),
    "gpt-4": ModelPricing(input_price_per_1k=0.03, output_price_per_1k=0.06),
    # GPT-3.5
    "gpt-3.5-turbo": ModelPricing(input_price_per_1k=0.0005, output_price_per_1k=0.0015),
    # Claude
    "claude-3-opus": ModelPricing(input_price_per_1k=0.015, output_price_per_1k=0.075),
    "claude-3-sonnet": ModelPricing(input_price_per_1k=0.003, output_price_per_1k=0.015),
    "claude-3-haiku": ModelPricing(input_price_per_1k=0.00025, output_price_per_1k=0.00125),
}

# Model to encoding mapping
MODEL_ENCODING: dict[str, str] = {
    "gpt-4o": "o200k_base",
    "gpt-4o-mini": "o200k_base",
    "gpt-4-turbo": "cl100k_base",
    "gpt-4": "cl100k_base",
    "gpt-3.5-turbo": "cl100k_base",
}


@lru_cache(maxsize=10)
def get_encoding(model: str) -> tiktoken.Encoding:
    """
    Get the tiktoken encoding for a model.

    Args:
        model: Model name.

    Returns:
        Tiktoken encoding.
    """
    encoding_name = MODEL_ENCODING.get(model, "cl100k_base")
    return tiktoken.get_encoding(encoding_name)


class TokenCounter:
    """Counts tokens for a specific model."""

    def __init__(self, model: str = "gpt-4o"):
        self.model = model
        self.encoding = get_encoding(model)

    def count(self, text: str) -> int:
        """
        Count tokens in text.

        Args:
            text: Text to count tokens for.

        Returns:
            Number of tokens.
        """
        if not text:
            return 0
        return len(self.encoding.encode(text))

    def count_messages(self, messages: list[dict[str, str]]) -> int:
        """
        Count tokens in a list of chat messages.

        Args:
            messages: List of message dictionaries.

        Returns:
            Total token count including message overhead.
        """
        # Message format overhead (role markers, etc.)
        # Based on OpenAI's documentation
        tokens_per_message = 3  # <|start|>{role}\n{content}<|end|>
        tokens_per_name = 1

        total = 0
        for message in messages:
            total += tokens_per_message
            for key, value in message.items():
                total += self.count(str(value))
                if key == "name":
                    total += tokens_per_name

        total += 3  # Reply priming
        return total


def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """
    Count tokens in text for a specific model.

    Args:
        text: Text to count.
        model: Model name.

    Returns:
        Token count.
    """
    counter = TokenCounter(model)
    return counter.count(text)


def count_message_tokens(
    messages: list[dict[str, str]],
    model: str = "gpt-4o",
) -> int:
    """
    Count tokens in chat messages.

    Args:
        messages: List of message dictionaries.
        model: Model name.

    Returns:
        Total token count.
    """
    counter = TokenCounter(model)
    return counter.count_messages(messages)


def get_model_pricing(model: str) -> ModelPricing | None:
    """
    Get pricing for a model.

    Args:
        model: Model name.

    Returns:
        Pricing info or None if unknown.
    """
    # Check exact match first
    if model in MODEL_PRICING:
        return MODEL_PRICING[model]

    # Check partial match (e.g., "gpt-4-turbo-2024-01-01" -> "gpt-4-turbo")
    for known_model, pricing in MODEL_PRICING.items():
        if model.startswith(known_model):
            return pricing

    return None


def estimate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> float:
    """
    Estimate cost for a request.

    Args:
        model: Model name.
        input_tokens: Number of input tokens.
        output_tokens: Number of output tokens.

    Returns:
        Estimated cost in USD.
    """
    pricing = get_model_pricing(model)
    if pricing is None:
        return 0.0

    input_cost = (input_tokens / 1000) * pricing.input_price_per_1k
    output_cost = (output_tokens / 1000) * pricing.output_price_per_1k

    return input_cost + output_cost


def check_context_limit(
    messages: list[dict[str, str]],
    model: str,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    """
    Check if messages fit within context limit.

    Args:
        messages: Chat messages.
        model: Model name.
        max_tokens: Override for max tokens.

    Returns:
        Dict with status and token info.
    """
    # Default context limits
    context_limits = {
        "gpt-4o": 128000,
        "gpt-4o-mini": 128000,
        "gpt-4-turbo": 128000,
        "gpt-4": 8192,
        "gpt-3.5-turbo": 16385,
    }

    token_count = count_message_tokens(messages, model)
    context_limit = max_tokens or context_limits.get(model, 8192)

    return {
        "token_count": token_count,
        "context_limit": context_limit,
        "remaining": context_limit - token_count,
        "fits": token_count <= context_limit,
    }
