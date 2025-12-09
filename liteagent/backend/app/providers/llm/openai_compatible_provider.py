"""
OpenAI-compatible provider for custom endpoints.
Supports Azure OpenAI, vLLM, LiteLLM proxy, and any OpenAI-compatible API.
"""
from typing import AsyncIterator

from openai import AsyncOpenAI

from app.providers.llm.base import BaseLLMProvider, LLMMessage, LLMResponse


class OpenAICompatibleProvider(BaseLLMProvider):
    """
    OpenAI-compatible API provider.

    Supports any endpoint that implements the OpenAI Chat Completions API:
    - Azure OpenAI
    - vLLM
    - LiteLLM proxy
    - LocalAI
    - Ollama (OpenAI compatibility mode)
    - Any other OpenAI-compatible server
    """

    def __init__(
        self,
        api_key: str,
        model_name: str,
        base_url: str | None = None,
        api_version: str | None = None,
        extra_headers: dict[str, str] | None = None,
        timeout: float = 120.0,
    ):
        super().__init__(api_key, model_name, base_url)
        self.api_version = api_version
        self.extra_headers = extra_headers or {}
        self.timeout = timeout

        # Build default headers
        default_headers = {}
        if extra_headers:
            default_headers.update(extra_headers)

        # Handle Azure OpenAI specifics
        if api_version and base_url and "azure" in base_url.lower():
            # Azure uses api-key header instead of Authorization Bearer
            default_headers["api-key"] = api_key

        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            default_headers=default_headers if default_headers else None,
            timeout=timeout,
        )

    async def chat(self, messages: list[LLMMessage]) -> LLMResponse:
        """Send messages to the LLM and get a response."""
        formatted_messages = [{"role": m.role, "content": m.content} for m in messages]

        # Build request kwargs
        request_kwargs = {
            "model": self.model_name,
            "messages": formatted_messages,
        }

        # Add API version for Azure
        if self.api_version:
            request_kwargs["extra_query"] = {"api-version": self.api_version}

        response = await self.client.chat.completions.create(**request_kwargs)

        return LLMResponse(
            content=response.choices[0].message.content or "",
            model=response.model,
            usage={
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0,
            },
        )

    async def stream_chat(self, messages: list[LLMMessage]) -> AsyncIterator[str]:
        """Stream responses from the LLM."""
        formatted_messages = [{"role": m.role, "content": m.content} for m in messages]

        request_kwargs = {
            "model": self.model_name,
            "messages": formatted_messages,
            "stream": True,
        }

        if self.api_version:
            request_kwargs["extra_query"] = {"api-version": self.api_version}

        stream = await self.client.chat.completions.create(**request_kwargs)

        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    @staticmethod
    def get_available_models() -> list[str]:
        """
        Return empty list since OpenAI-compatible endpoints have user-defined models.
        Users provide their own model/deployment names.
        """
        return []
