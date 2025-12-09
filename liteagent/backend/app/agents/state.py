"""
Agent state management.

Contains AgentState for managing agent execution state.
Implements Factor 5 (Unified execution and business state) and Factor 6 (Launch/Pause/Resume).
"""
import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from app.agents.types import AgentStatus, AgentStep, StepType


@dataclass
class AgentState:
    """
    Complete agent state - can be serialized and restored.
    Factor 5: Unified execution and business state.
    Factor 6: Enables pause/resume.
    Factor 12: Input to stateless reducer.
    """
    agent_id: str
    status: AgentStatus = AgentStatus.IDLE
    steps: list[AgentStep] = field(default_factory=list)
    current_iteration: int = 0
    max_iterations: int = 10
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_step(self, step: AgentStep) -> None:
        """Add a step to execution history."""
        self.steps.append(step)

    def get_context_messages(self) -> list[dict[str, Any]]:
        """
        Factor 3: Own your context window.
        Convert steps to LLM message format.
        """
        messages = []
        for step in self.steps:
            if step.step_type == StepType.USER_MESSAGE:
                messages.append({"role": "user", "content": step.content})
            elif step.step_type == StepType.ASSISTANT_MESSAGE:
                messages.append({"role": "assistant", "content": step.content})
            elif step.step_type == StepType.TOOL_CALL:
                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [step.content],
                })
            elif step.step_type == StepType.TOOL_RESULT:
                messages.append({
                    "role": "tool",
                    "tool_call_id": step.content.get("tool_call_id"),
                    "content": json.dumps(step.content.get("result", {})),
                })
            elif step.step_type == StepType.ERROR:
                # Factor 9: Compact errors into context
                messages.append({
                    "role": "tool",
                    "tool_call_id": step.content.get("tool_call_id", "error"),
                    "content": f"Error: {step.content.get('error', 'Unknown error')}",
                })
        return messages

    def to_dict(self) -> dict[str, Any]:
        """Serialize state for persistence."""
        return {
            "agent_id": self.agent_id,
            "status": self.status.value,
            "steps": [s.to_dict() for s in self.steps],
            "current_iteration": self.current_iteration,
            "max_iterations": self.max_iterations,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentState":
        """Deserialize state for resume."""
        state = cls(
            agent_id=data["agent_id"],
            status=AgentStatus(data["status"]),
            current_iteration=data["current_iteration"],
            max_iterations=data["max_iterations"],
            metadata=data.get("metadata", {}),
        )
        for step_data in data.get("steps", []):
            state.steps.append(AgentStep(
                step_type=StepType(step_data["step_type"]),
                content=step_data["content"],
                metadata=step_data.get("metadata", {}),
            ))
        return state

    def compute_hash(self) -> str:
        """Compute hash for state comparison."""
        content = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha256(content.encode()).hexdigest()[:16]
