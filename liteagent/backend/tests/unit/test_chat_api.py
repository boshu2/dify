"""
Tests for Chat API endpoints.

Tests streaming and non-streaming chat endpoints.
"""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient
from httpx import AsyncClient, ASGITransport
import json

from app.api.routes.chat import (
    router,
    ChatRequest,
    ChatResponse,
    MockStreamingLLMClient,
    create_agent_for_chat,
    stream_response,
)


class TestChatRequest:
    """Test ChatRequest model."""

    def test_default_values(self):
        """Request should have sensible defaults."""
        request = ChatRequest(message="Hello")

        assert request.agent_id == "default"
        assert request.message == "Hello"
        assert request.conversation_history == []
        assert request.system_prompt == "You are a helpful assistant."

    def test_custom_values(self):
        """Request should accept custom values."""
        history = [{"role": "user", "content": "Hi"}]
        request = ChatRequest(
            agent_id="custom-agent",
            message="Hello there",
            conversation_history=history,
            system_prompt="You are a pirate.",
        )

        assert request.agent_id == "custom-agent"
        assert request.message == "Hello there"
        assert request.conversation_history == history
        assert request.system_prompt == "You are a pirate."


class TestChatResponse:
    """Test ChatResponse model."""

    def test_response_creation(self):
        """Response should contain all fields."""
        response = ChatResponse(
            response="Hello!",
            status="completed",
            steps=3,
        )

        assert response.response == "Hello!"
        assert response.status == "completed"
        assert response.steps == 3


class TestMockStreamingLLMClient:
    """Test MockStreamingLLMClient."""

    @pytest.mark.asyncio
    async def test_chat_returns_response(self):
        """Chat should return a response based on user message."""
        client = MockStreamingLLMClient()

        messages = [
            {"role": "user", "content": "Hello, how are you?"},
        ]

        result = await client.chat(messages)

        assert "choices" in result
        assert len(result["choices"]) == 1
        assert "message" in result["choices"][0]
        assert "content" in result["choices"][0]["message"]
        assert "Hello, how are you?" in result["choices"][0]["message"]["content"]

    @pytest.mark.asyncio
    async def test_chat_extracts_last_user_message(self):
        """Chat should use the last user message."""
        client = MockStreamingLLMClient()

        messages = [
            {"role": "user", "content": "First message"},
            {"role": "assistant", "content": "Response"},
            {"role": "user", "content": "Second message"},
        ]

        result = await client.chat(messages)

        content = result["choices"][0]["message"]["content"]
        assert "Second message" in content
        assert "First message" not in content

    @pytest.mark.asyncio
    async def test_chat_handles_empty_messages(self):
        """Chat should handle empty message list."""
        client = MockStreamingLLMClient()

        result = await client.chat([])

        assert "choices" in result


class TestCreateAgentForChat:
    """Test create_agent_for_chat factory."""

    def test_creates_agent(self):
        """Factory should create a functional agent."""
        agent = create_agent_for_chat(
            agent_id="test-agent",
            system_prompt="You are helpful.",
            purpose="Test purpose",
        )

        assert agent is not None
        assert agent.config.agent_id == "test-agent"
        assert agent.config.purpose == "Test purpose"

    def test_agent_has_llm_client(self):
        """Agent should have an LLM client."""
        agent = create_agent_for_chat(
            agent_id="test",
            system_prompt="Test",
        )

        assert agent.config.llm_client is not None
        assert isinstance(agent.config.llm_client, MockStreamingLLMClient)


class TestStreamResponse:
    """Test stream_response generator."""

    @pytest.mark.asyncio
    async def test_streams_content_chunks(self):
        """Should stream content in chunks."""
        text = "Hello world this is a test"
        chunks = []

        async for chunk in stream_response(text, chunk_size=2):
            chunks.append(chunk)

        # Should have content chunks plus done event
        assert len(chunks) >= 2

        # Last chunk should be done event
        assert '"type": "done"' in chunks[-1]

    @pytest.mark.asyncio
    async def test_sse_format(self):
        """Chunks should be in SSE format."""
        text = "Hello world"

        async for chunk in stream_response(text, chunk_size=10):
            assert chunk.startswith("data: ")
            assert chunk.endswith("\n\n")

            # Should be valid JSON after "data: "
            json_str = chunk[6:-2]  # Remove "data: " and "\n\n"
            data = json.loads(json_str)
            assert "type" in data

    @pytest.mark.asyncio
    async def test_content_reconstructs_original(self):
        """Concatenated content should approximate original text."""
        text = "Hello world this is a test message"
        content_parts = []

        async for chunk in stream_response(text, chunk_size=2):
            data = json.loads(chunk[6:-2])
            if data["type"] == "content":
                content_parts.append(data["content"])

        reconstructed = "".join(content_parts)
        # Check that all words are present
        for word in text.split():
            assert word in reconstructed


class TestChatEndpoints:
    """Test chat API endpoints using TestClient."""

    @pytest.fixture
    def app(self):
        """Create test app with chat router."""
        from fastapi import FastAPI
        app = FastAPI()
        app.include_router(router)
        return app

    @pytest.fixture
    def client(self, app):
        """Create test client."""
        return TestClient(app)

    def test_health_endpoint(self, client):
        """Health endpoint should return status."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "chat"
        assert data["features"]["streaming"] is True
        assert data["features"]["twelve_factor_agent"] is True

    def test_stream_endpoint_requires_message(self, client):
        """Stream endpoint should reject empty message."""
        response = client.post("/stream", json={"message": ""})

        assert response.status_code == 400

    def test_stream_endpoint_returns_sse(self, client):
        """Stream endpoint should return SSE stream."""
        response = client.post(
            "/stream",
            json={"message": "Hello"},
        )

        assert response.status_code == 200
        assert response.headers["content-type"] == "text/event-stream; charset=utf-8"

    def test_stream_endpoint_sends_events(self, client):
        """Stream endpoint should send SSE events."""
        response = client.post(
            "/stream",
            json={"message": "Hello"},
        )

        content = response.text

        # Should have content events
        assert "data: " in content
        assert '"type": "content"' in content

        # Should end with done event
        assert '"type": "done"' in content

    def test_complete_endpoint_requires_message(self, client):
        """Complete endpoint should reject empty message."""
        response = client.post("/complete", json={"message": ""})

        assert response.status_code == 400

    def test_complete_endpoint_returns_response(self, client):
        """Complete endpoint should return chat response."""
        response = client.post(
            "/complete",
            json={"message": "Hello"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "response" in data
        assert "status" in data
        assert "steps" in data
        assert data["steps"] >= 1

    def test_agent_state_endpoint(self, client):
        """Agent state endpoint should return state info."""
        response = client.get("/agents/test-agent/state")

        assert response.status_code == 200
        data = response.json()
        assert data["agent_id"] == "test-agent"
        assert "status" in data


class TestChatEndpointsAsync:
    """Async tests for chat endpoints."""

    @pytest.fixture
    def app(self):
        """Create test app with chat router."""
        from fastapi import FastAPI
        app = FastAPI()
        app.include_router(router)
        return app

    @pytest.mark.asyncio
    async def test_stream_full_flow(self, app):
        """Test complete streaming flow."""
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test"
        ) as client:
            response = await client.post(
                "/stream",
                json={
                    "agent_id": "test-agent",
                    "message": "What is 2+2?",
                    "system_prompt": "You are a math tutor.",
                },
            )

            assert response.status_code == 200

            # Parse all events
            events = []
            for line in response.text.split("\n"):
                if line.startswith("data: "):
                    events.append(json.loads(line[6:]))

            # Should have at least content and done events
            event_types = [e["type"] for e in events]
            assert "content" in event_types
            assert "done" in event_types

    @pytest.mark.asyncio
    async def test_complete_with_history(self, app):
        """Test complete endpoint with conversation history."""
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test"
        ) as client:
            response = await client.post(
                "/complete",
                json={
                    "agent_id": "test-agent",
                    "message": "And what about 3+3?",
                    "conversation_history": [
                        {"role": "user", "content": "What is 2+2?"},
                        {"role": "assistant", "content": "2+2 equals 4."},
                    ],
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "completed"


class TestChatIntegration:
    """Integration tests for chat with 12-factor agent."""

    @pytest.mark.asyncio
    async def test_agent_completes_chat(self):
        """Agent should complete a chat interaction."""
        from app.agents.twelve_factor_agent import AgentStatus

        agent = create_agent_for_chat(
            agent_id="integration-test",
            system_prompt="You are a helpful assistant.",
        )

        state = agent.launch("Hello!")
        state = await agent.run_to_completion(state)

        assert state.status == AgentStatus.COMPLETED
        assert len(state.steps) >= 2  # User message + assistant response

    @pytest.mark.asyncio
    async def test_agent_handles_multiple_turns(self):
        """Agent should handle multi-turn conversations."""
        from app.agents.twelve_factor_agent import AgentStatus

        agent = create_agent_for_chat(
            agent_id="multi-turn-test",
            system_prompt="You are a helpful assistant.",
        )

        # First turn
        state = agent.launch("Hello!")
        state = await agent.run_to_completion(state)
        assert state.status == AgentStatus.COMPLETED

        # Second turn (new state, simulating continuation)
        state2 = agent.launch("How are you?")
        state2 = await agent.run_to_completion(state2)
        assert state2.status == AgentStatus.COMPLETED
