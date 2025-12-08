"""
Agent type definitions.

Contains enums and data classes used throughout the agent system.
"""
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable


class AgentStatus(str, Enum):
    """Agent execution status."""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"  # Factor 6: Pause capability
    WAITING_HUMAN = "waiting_human"  # Factor 7: Human contact
    COMPLETED = "completed"
    FAILED = "failed"


class StepType(str, Enum):
    """Types of agent steps."""
    USER_MESSAGE = "user_message"
    ASSISTANT_MESSAGE = "assistant_message"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    HUMAN_REQUEST = "human_request"  # Factor 7
    HUMAN_RESPONSE = "human_response"
    ERROR = "error"


@dataclass
class AgentStep:
    """
    A single step in agent execution.
    Factor 5: All execution state is captured in steps.
    """
    step_type: StepType
    content: Any
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_type": self.step_type.value,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class ToolDefinition:
    """
    Factor 4: Tools are structured outputs.
    Definition of a tool the agent can call.
    """
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., Any] | None = None
    requires_human_approval: bool = False  # Factor 7

    def to_openai_format(self) -> dict[str, Any]:
        """Convert to OpenAI function format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass
class HumanContactRequest:
    """
    Factor 7: Contact humans with tool calls.
    Request for human input/approval.
    """
    request_id: str
    request_type: str  # "approval", "input", "clarification"
    message: str
    context: dict[str, Any] = field(default_factory=dict)
    timeout_seconds: int = 3600
