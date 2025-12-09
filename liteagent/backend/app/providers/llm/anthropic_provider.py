from typing import AsyncIterator

from anthropic import AsyncAnthropic

from app.providers.llm.base import BaseLLMProvider, LLMMessage, LLMResponse


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude API provider."""

    AVAILABLE_MODELS = [
        "claude-opus-4-5-20251101",
        "claude-sonnet-4-5-20250929",
        "claude-3-5-sonnet-20241022",
        "claude-3-5-haiku-20241022",
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "claude-3-haiku-20240307",
    ]

    def __init__(self, api_key: str, model_name: str, base_url: str | None = None):
        super().__init__(api_key, model_name, base_url)
        self.client = AsyncAnthropic(
            api_key=api_key,
            base_url=base_url if base_url else None,
        )

    async def chat(self, messages: list[LLMMessage]) -> LLMResponse:
        # Extract system message if present
        system_message = None
        chat_messages = []

        for msg in messages:
            if msg.role == "system":
                system_message = msg.content
            else:
                chat_messages.append({"role": msg.role, "content": msg.content})

        response = await self.client.messages.create(
            model=self.model_name,
            max_tokens=4096,
            system=system_message if system_message else "",
            messages=chat_messages,
        )

        return LLMResponse(
            content=response.content[0].text if response.content else "",
            model=response.model,
            usage={
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
            },
        )

    async def stream_chat(self, messages: list[LLMMessage]) -> AsyncIterator[str]:
        # Extract system message if present
        system_message = None
        chat_messages = []

        for msg in messages:
            if msg.role == "system":
                system_message = msg.content
            else:
                chat_messages.append({"role": msg.role, "content": msg.content})

        async with self.client.messages.stream(
            model=self.model_name,
            max_tokens=4096,
            system=system_message if system_message else "",
            messages=chat_messages,
        ) as stream:
            async for text in stream.text_stream:
                yield text

    @staticmethod
    def get_available_models() -> list[str]:
        return AnthropicProvider.AVAILABLE_MODELS
