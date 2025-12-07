"""
Server-Sent Events (SSE) streaming support for LiteAgent.
Provides utilities for streaming chat responses to clients.
"""
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator

from starlette.responses import StreamingResponse as StarletteStreamingResponse


class StreamEventType(str, Enum):
    """Types of streaming events."""

    TEXT = "text"
    METADATA = "metadata"
    ERROR = "error"
    DONE = "done"


@dataclass
class StreamEvent:
    """A streaming event to send to the client."""

    event_type: StreamEventType
    data: Any = None
    id: str | None = None


def format_sse_event(event: StreamEvent) -> str:
    """
    Format a stream event as an SSE message.

    Args:
        event: The event to format.

    Returns:
        SSE-formatted string.
    """
    lines = []

    if event.id:
        lines.append(f"id: {event.id}")

    lines.append(f"event: {event.event_type.value}")

    # Serialize data to JSON
    if event.data is None:
        data_str = "null"
    elif isinstance(event.data, str):
        data_str = json.dumps(event.data)
    else:
        data_str = json.dumps(event.data)

    lines.append(f"data: {data_str}")

    return "\n".join(lines) + "\n\n"


def parse_sse_event(sse_text: str) -> StreamEvent:
    """
    Parse an SSE message into a StreamEvent.

    Args:
        sse_text: The SSE-formatted text.

    Returns:
        Parsed StreamEvent.
    """
    event_type = StreamEventType.TEXT
    data = None
    event_id = None

    for line in sse_text.strip().split("\n"):
        if line.startswith("event: "):
            event_type = StreamEventType(line[7:])
        elif line.startswith("data: "):
            data_str = line[6:]
            data = json.loads(data_str)
        elif line.startswith("id: "):
            event_id = line[4:]

    return StreamEvent(event_type=event_type, data=data, id=event_id)


class StreamingResponse(StarletteStreamingResponse):
    """
    Streaming HTTP response for SSE.

    Wraps an async generator of StreamEvents and formats them as SSE.
    """

    def __init__(
        self,
        event_generator: AsyncIterator[StreamEvent],
        status_code: int = 200,
        headers: dict[str, str] | None = None,
    ):
        """
        Initialize streaming response.

        Args:
            event_generator: Async generator yielding StreamEvents.
            status_code: HTTP status code.
            headers: Additional HTTP headers.
        """
        self._event_generator = event_generator

        # Set SSE headers
        sse_headers = {
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable Nginx buffering
        }
        if headers:
            sse_headers.update(headers)

        super().__init__(
            content=self._stream_events(),
            status_code=status_code,
            headers=sse_headers,
            media_type="text/event-stream",
        )

    async def _stream_events(self) -> AsyncIterator[bytes]:
        """Stream formatted SSE events."""
        async for event in self._event_generator:
            yield format_sse_event(event).encode("utf-8")
