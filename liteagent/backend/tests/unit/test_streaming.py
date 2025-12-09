"""
Unit tests for streaming chat support.
Tests Server-Sent Events streaming and streaming LLM responses.
"""
import json
import pytest
from unittest.mock import Mock, AsyncMock, patch
from typing import AsyncIterator

from app.core.streaming import (
    StreamEvent,
    StreamEventType,
    StreamingResponse,
    format_sse_event,
    parse_sse_event,
)
from app.providers.llm.streaming import (
    StreamingChatHandler,
    StreamChunk,
)


class TestStreamEvent:
    """Tests for stream event data structure."""

    def test_create_text_event(self):
        """Test creating a text chunk event."""
        event = StreamEvent(
            event_type=StreamEventType.TEXT,
            data="Hello",
        )
        assert event.event_type == StreamEventType.TEXT
        assert event.data == "Hello"

    def test_create_done_event(self):
        """Test creating a done event."""
        event = StreamEvent(
            event_type=StreamEventType.DONE,
            data=None,
        )
        assert event.event_type == StreamEventType.DONE

    def test_create_error_event(self):
        """Test creating an error event."""
        event = StreamEvent(
            event_type=StreamEventType.ERROR,
            data="Connection failed",
        )
        assert event.event_type == StreamEventType.ERROR
        assert event.data == "Connection failed"

    def test_create_metadata_event(self):
        """Test creating a metadata event with token counts."""
        event = StreamEvent(
            event_type=StreamEventType.METADATA,
            data={"prompt_tokens": 10, "completion_tokens": 20},
        )
        assert event.event_type == StreamEventType.METADATA
        assert event.data["prompt_tokens"] == 10


class TestSSEFormatting:
    """Tests for SSE message formatting."""

    def test_format_text_event(self):
        """Test formatting text event as SSE."""
        event = StreamEvent(event_type=StreamEventType.TEXT, data="Hello")
        sse = format_sse_event(event)

        assert "event: text" in sse
        assert 'data: "Hello"' in sse
        assert sse.endswith("\n\n")

    def test_format_done_event(self):
        """Test formatting done event as SSE."""
        event = StreamEvent(event_type=StreamEventType.DONE, data=None)
        sse = format_sse_event(event)

        assert "event: done" in sse
        assert "data: null" in sse

    def test_format_metadata_event(self):
        """Test formatting metadata event as SSE."""
        event = StreamEvent(
            event_type=StreamEventType.METADATA,
            data={"tokens": 100},
        )
        sse = format_sse_event(event)

        assert "event: metadata" in sse
        assert '"tokens": 100' in sse

    def test_parse_text_event(self):
        """Test parsing SSE text event."""
        sse = 'event: text\ndata: "Hello"\n\n'
        event = parse_sse_event(sse)

        assert event.event_type == StreamEventType.TEXT
        assert event.data == "Hello"

    def test_parse_metadata_event(self):
        """Test parsing SSE metadata event."""
        sse = 'event: metadata\ndata: {"tokens": 100}\n\n'
        event = parse_sse_event(sse)

        assert event.event_type == StreamEventType.METADATA
        assert event.data["tokens"] == 100


class TestStreamChunk:
    """Tests for streaming chunk data structure."""

    def test_create_content_chunk(self):
        """Test creating a content chunk."""
        chunk = StreamChunk(
            content="Hello",
            is_final=False,
        )
        assert chunk.content == "Hello"
        assert chunk.is_final is False

    def test_create_final_chunk(self):
        """Test creating a final chunk."""
        chunk = StreamChunk(
            content="",
            is_final=True,
            usage={"prompt_tokens": 10, "completion_tokens": 20},
        )
        assert chunk.is_final is True
        assert chunk.usage["completion_tokens"] == 20


class TestStreamingChatHandler:
    """Tests for streaming chat handler."""

    @pytest.mark.asyncio
    async def test_stream_chunks(self):
        """Test streaming chunks from a generator."""
        async def mock_generator() -> AsyncIterator[StreamChunk]:
            yield StreamChunk(content="Hello", is_final=False)
            yield StreamChunk(content=" World", is_final=False)
            yield StreamChunk(content="", is_final=True)

        handler = StreamingChatHandler()
        chunks = []
        async for chunk in handler.process_stream(mock_generator()):
            chunks.append(chunk)

        assert len(chunks) == 3
        assert chunks[0].content == "Hello"
        assert chunks[1].content == " World"
        assert chunks[2].is_final is True

    @pytest.mark.asyncio
    async def test_accumulate_response(self):
        """Test accumulating full response from chunks."""
        async def mock_generator() -> AsyncIterator[StreamChunk]:
            yield StreamChunk(content="Hello", is_final=False)
            yield StreamChunk(content=" ", is_final=False)
            yield StreamChunk(content="World", is_final=False)
            yield StreamChunk(content="", is_final=True)

        handler = StreamingChatHandler()
        full_response = await handler.get_full_response(mock_generator())

        assert full_response == "Hello World"

    @pytest.mark.asyncio
    async def test_handle_stream_error(self):
        """Test handling errors during streaming."""
        async def error_generator() -> AsyncIterator[StreamChunk]:
            yield StreamChunk(content="Hello", is_final=False)
            raise RuntimeError("Stream error")

        handler = StreamingChatHandler()

        with pytest.raises(RuntimeError, match="Stream error"):
            async for _ in handler.process_stream(error_generator()):
                pass


class TestStreamingResponse:
    """Tests for streaming HTTP response."""

    @pytest.mark.asyncio
    async def test_create_streaming_response(self):
        """Test creating a streaming SSE response."""
        events = [
            StreamEvent(event_type=StreamEventType.TEXT, data="Hello"),
            StreamEvent(event_type=StreamEventType.DONE, data=None),
        ]

        async def event_generator():
            for event in events:
                yield event

        response = StreamingResponse(event_generator())

        assert response.media_type == "text/event-stream"
        assert response.headers.get("Cache-Control") == "no-cache"
        assert response.headers.get("Connection") == "keep-alive"

    @pytest.mark.asyncio
    async def test_iterate_streaming_response(self):
        """Test iterating over streaming response."""
        events = [
            StreamEvent(event_type=StreamEventType.TEXT, data="Hello"),
            StreamEvent(event_type=StreamEventType.TEXT, data=" World"),
            StreamEvent(event_type=StreamEventType.DONE, data=None),
        ]

        async def event_generator():
            for event in events:
                yield event

        response = StreamingResponse(event_generator())
        collected = []

        async for chunk in response.body_iterator:
            collected.append(chunk.decode() if isinstance(chunk, bytes) else chunk)

        assert len(collected) == 3
        assert "Hello" in collected[0]
        assert "World" in collected[1]
        assert "done" in collected[2]
