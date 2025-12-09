from typing import AsyncIterator

import httpx

from app.providers.llm.base import BaseLLMProvider, LLMMessage, LLMResponse


class OllamaProvider(BaseLLMProvider):
    """Ollama local LLM provider."""

    DEFAULT_MODELS = [
        "llama3.2",
        "llama3.1",
        "llama2",
        "codellama",
        "mistral",
        "mixtral",
        "phi3",
        "qwen2",
    ]

    def __init__(self, api_key: str | None, model_name: str, base_url: str | None = None):
        super().__init__(api_key, model_name, base_url)
        self.base_url = base_url or "http://localhost:11434"

    async def chat(self, messages: list[LLMMessage]) -> LLMResponse:
        formatted_messages = [{"role": m.role, "content": m.content} for m in messages]

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model_name,
                    "messages": formatted_messages,
                    "stream": False,
                },
            )
            response.raise_for_status()
            data = response.json()

        return LLMResponse(
            content=data.get("message", {}).get("content", ""),
            model=self.model_name,
            usage={
                "prompt_tokens": data.get("prompt_eval_count", 0),
                "completion_tokens": data.get("eval_count", 0),
                "total_tokens": data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
            },
        )

    async def stream_chat(self, messages: list[LLMMessage]) -> AsyncIterator[str]:
        formatted_messages = [{"role": m.role, "content": m.content} for m in messages]

        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model_name,
                    "messages": formatted_messages,
                    "stream": True,
                },
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line:
                        import json
                        data = json.loads(line)
                        if content := data.get("message", {}).get("content"):
                            yield content

    @staticmethod
    def get_available_models() -> list[str]:
        return OllamaProvider.DEFAULT_MODELS

    async def list_local_models(self) -> list[str]:
        """Fetch actually installed models from Ollama."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                response.raise_for_status()
                data = response.json()
                return [model["name"] for model in data.get("models", [])]
        except Exception:
            return self.DEFAULT_MODELS
