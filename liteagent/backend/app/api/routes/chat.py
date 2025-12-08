"""
Chat API endpoints with streaming support.

Integrates with the 12-factor agent for conversational AI.
"""
import asyncio
import json
from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.agents.twelve_factor_agent import (
    Agent,
    AgentConfig,
    LLMClient,
    StepType,
)

router = APIRouter()


class ChatRequest(BaseModel):
    """Request body for chat endpoints."""
    agent_id: str = "default"
    message: str
    conversation_history: list[dict] = []
    system_prompt: str = "You are a helpful assistant."


class ChatResponse(BaseModel):
    """Response body for non-streaming chat."""
    response: str
    status: str
    steps: int


class MockStreamingLLMClient(LLMClient):
    """
    Mock LLM client that simulates responses.
    Replace with real OpenAI/Anthropic client in production.
    """

    def __init__(self, api_key: str | None = None, model: str = "gpt-4"):
        self.api_key = api_key
        self.model = model

    async def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> dict:
        """Non-streaming chat for tool calls."""
        last_user_message = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                last_user_message = msg.get("content", "")
                break

        # Simulate response
        user_excerpt = last_user_message[:100]
        response_content = f"I understand you said: '{user_excerpt}'. How can I help?"

        return {
            "choices": [{
                "message": {
                    "content": response_content,
                },
            }],
        }


def create_agent_for_chat(
    agent_id: str,
    system_prompt: str,
    purpose: str = "Chat assistant",
) -> Agent:
    """Create a 12-factor agent for chat."""
    config = AgentConfig(
        agent_id=agent_id,
        purpose=purpose,
        tools=[],
        max_iterations=5,
        llm_client=MockStreamingLLMClient(),
    )
    return Agent(config)


async def stream_response(text: str, chunk_size: int = 3) -> AsyncGenerator[str, None]:
    """
    Stream a response in chunks, simulating token-by-token generation.

    In production, this would stream from the actual LLM API.
    """
    words = text.split()

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        if i > 0:
            chunk = " " + chunk

        # SSE format
        data = {
            "type": "content",
            "content": chunk,
        }
        yield f"data: {json.dumps(data)}\n\n"
        await asyncio.sleep(0.05)  # Simulate generation delay

    # Send completion event
    yield f"data: {json.dumps({'type': 'done'})}\n\n"


@router.post("/stream")
async def stream_chat(request: ChatRequest) -> StreamingResponse:
    """
    Streaming chat endpoint using Server-Sent Events (SSE).

    Request body:
    {
        "agent_id": "agent-123",
        "message": "Hello, how are you?",
        "conversation_history": [...],
        "system_prompt": "You are a helpful assistant."
    }

    Response: SSE stream with events:
    - data: {"type": "content", "content": "token..."}
    - data: {"type": "tool_call", "tool": "...", "args": {...}}
    - data: {"type": "tool_result", "result": {...}}
    - data: {"type": "done"}
    - data: {"type": "error", "message": "..."}
    """
    if not request.message:
        raise HTTPException(status_code=400, detail="Message is required")

    # For demo, generate a simple response
    response_text = (
        f"Thank you for your message. You said: '{request.message}'. "
        "I'm a 12-factor agent ready to help you with various tasks. "
        "I can search knowledge bases, execute tools, and maintain conversation context. "
        "What would you like to know?"
    )

    async def generate() -> AsyncGenerator[str, None]:
        try:
            async for chunk in stream_response(response_text):
                yield chunk
        except Exception as e:
            error_data = {"type": "error", "message": str(e)}
            yield f"data: {json.dumps(error_data)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/complete", response_model=ChatResponse)
async def complete_chat(request: ChatRequest) -> ChatResponse:
    """
    Non-streaming chat completion endpoint.

    Request body:
    {
        "agent_id": "agent-123",
        "message": "Hello",
        "conversation_history": [...],
        "system_prompt": "..."
    }

    Response:
    {
        "response": "Assistant's response",
        "status": "completed",
        "steps": 2
    }
    """
    if not request.message:
        raise HTTPException(status_code=400, detail="Message is required")

    # Create and run agent
    agent = create_agent_for_chat(
        agent_id=request.agent_id,
        system_prompt=request.system_prompt,
    )

    state = agent.launch(request.message)
    state = await agent.run_to_completion(state)

    # Extract response
    response_text = ""
    for step in reversed(state.steps):
        if step.step_type == StepType.ASSISTANT_MESSAGE:
            response_text = step.content
            break

    return ChatResponse(
        response=response_text,
        status=state.status.value,
        steps=len(state.steps),
    )


@router.get("/agents/{agent_id}/state")
async def get_agent_state(agent_id: str) -> dict:
    """
    Get the current state of an agent conversation.

    Useful for resuming conversations (Factor 6: Pause/Resume).
    """
    # In production, retrieve from database
    return {
        "agent_id": agent_id,
        "status": "idle",
        "message": "State retrieval not implemented - use database in production",
    }


@router.get("/health")
async def health() -> dict:
    """Health check for chat service."""
    return {
        "status": "healthy",
        "service": "chat",
        "features": {
            "streaming": True,
            "twelve_factor_agent": True,
            "rag_enabled": False,
        },
    }
