from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator


@dataclass
class LLMMessage:
    role: str  # system, user, assistant
    content: str


@dataclass
class LLMResponse:
    content: str
    model: str
    usage: dict | None = None


class BaseLLMProvider(ABC):
    """Base class for all LLM providers."""

    def __init__(self, api_key: str | None, model_name: str, base_url: str | None = None):
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = base_url

    @abstractmethod
    async def chat(self, messages: list[LLMMessage]) -> LLMResponse:
        """Send messages to the LLM and get a response."""
        pass

    @abstractmethod
    async def stream_chat(self, messages: list[LLMMessage]) -> AsyncIterator[str]:
        """Stream responses from the LLM."""
        pass

    @staticmethod
    def get_available_models() -> list[str]:
        """Return list of available models for this provider."""
        return []
