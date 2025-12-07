"""
Streaming support for LLM providers.
Handles processing of streaming responses from LLM APIs.
"""
from dataclasses import dataclass, field
from typing import Any, AsyncIterator


@dataclass
class StreamChunk:
    """A chunk of streaming content from an LLM."""

    content: str
    is_final: bool = False
    usage: dict[str, int] | None = None
    finish_reason: str | None = None


class StreamingChatHandler:
    """Handler for processing streaming chat responses."""

    def __init__(self):
        self._accumulated_content = ""

    async def process_stream(
        self, stream: AsyncIterator[StreamChunk]
    ) -> AsyncIterator[StreamChunk]:
        """
        Process a stream of chunks.

        Args:
            stream: Async iterator of StreamChunks.

        Yields:
            Each StreamChunk as it's received.
        """
        async for chunk in stream:
            self._accumulated_content += chunk.content
            yield chunk

    async def get_full_response(self, stream: AsyncIterator[StreamChunk]) -> str:
        """
        Consume a stream and return the full response.

        Args:
            stream: Async iterator of StreamChunks.

        Returns:
            The complete response text.
        """
        content = ""
        async for chunk in stream:
            content += chunk.content
        return content

    def reset(self) -> None:
        """Reset the accumulated content."""
        self._accumulated_content = ""

    @property
    def accumulated_content(self) -> str:
        """Get the accumulated content so far."""
        return self._accumulated_content
