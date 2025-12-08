"""
12-Factor Agent Implementation.

Based on https://github.com/humanlayer/12-factor-agents

The 12 Factors:
1. Natural Language to Tool Calls - LLM converts intent to structured calls
2. Own Your Prompts - Full control over prompt templates
3. Own Your Context Window - Deliberate context construction
4. Tools are Structured Outputs - Tool calls as constrained output format
5. Unify Execution & Business State - State inferable from context
6. Launch/Pause/Resume - Simple lifecycle management
7. Contact Humans with Tool Calls - Human-in-the-loop via tools
8. Own Your Control Flow - Explicit state transitions
9. Compact Errors into Context - Efficient error representation
10. Small, Focused Agents - Single-purpose agents
11. Trigger from Anywhere - Multiple entry points
12. Stateless Reducer - Pure function transforming state
"""
import hashlib
import json
from abc import ABC, abstractmethod
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


class AgentPrompts:
    """
    Factor 2: Own your prompts.
    All prompts in one place, fully controllable.
    """

    @staticmethod
    def system_prompt(agent_purpose: str, tools: list[ToolDefinition]) -> str:
        tool_descriptions = "\n".join(
            f"- {t.name}: {t.description}" for t in tools
        )
        return f"""You are an AI agent with a specific purpose: {agent_purpose}

Available tools:
{tool_descriptions}

Instructions:
1. Analyze the user's request carefully
2. Break down complex tasks into steps
3. Use tools when needed to accomplish tasks
4. If you need human approval or input, use the request_human_input tool
5. When the task is complete, provide a final response without tool calls

Always think step-by-step before acting."""

    @staticmethod
    def error_context(error: str, tool_name: str) -> str:
        """
        Factor 9: Compact errors into context.
        Keep error messages concise but informative.
        """
        return f"Tool '{tool_name}' failed: {error[:200]}"  # Truncate long errors


class LLMClient(ABC):
    """Abstract LLM client interface."""

    @abstractmethod
    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Send chat request to LLM."""
        pass


@dataclass
class AgentConfig:
    """Configuration for an agent instance."""
    agent_id: str
    purpose: str  # Factor 10: Small, focused agents
    tools: list[ToolDefinition] = field(default_factory=list)
    max_iterations: int = 10
    llm_client: LLMClient | None = None


class Agent:
    """
    12-Factor Agent Implementation.
    Factor 12: Agent as stateless reducer - transforms state, doesn't hold it.
    """

    def __init__(self, config: AgentConfig):
        self.config = config
        self._tool_handlers: dict[str, Callable] = {}

        # Register tool handlers
        for tool in config.tools:
            if tool.handler:
                self._tool_handlers[tool.name] = tool.handler

        # Factor 7: Built-in human contact tool
        self._register_human_contact_tool()

    def _register_human_contact_tool(self) -> None:
        """Factor 7: Add human contact capability."""
        human_tool = ToolDefinition(
            name="request_human_input",
            description="Request input or approval from a human operator",
            parameters={
                "type": "object",
                "properties": {
                    "request_type": {
                        "type": "string",
                        "enum": ["approval", "input", "clarification"],
                        "description": "Type of human interaction needed",
                    },
                    "message": {
                        "type": "string",
                        "description": "Message to show the human",
                    },
                },
                "required": ["request_type", "message"],
            },
        )
        self.config.tools.append(human_tool)

    def _get_system_prompt(self) -> str:
        """Factor 2: Full control over prompts."""
        return AgentPrompts.system_prompt(self.config.purpose, self.config.tools)

    def _get_tools_for_llm(self) -> list[dict[str, Any]]:
        """Factor 4: Tools as structured outputs."""
        return [t.to_openai_format() for t in self.config.tools]

    async def step(self, state: AgentState) -> AgentState:
        """
        Factor 12: Stateless reducer.
        Takes state, returns new state. No side effects stored in agent.
        Factor 8: Own your control flow.
        """
        if state.status in [AgentStatus.COMPLETED, AgentStatus.FAILED]:
            return state

        if state.status == AgentStatus.WAITING_HUMAN:
            # Can't proceed without human response
            return state

        if state.current_iteration >= state.max_iterations:
            state.status = AgentStatus.FAILED
            state.add_step(AgentStep(
                step_type=StepType.ERROR,
                content={"error": "Max iterations reached"},
            ))
            return state

        # Build messages for LLM
        # Factor 3: Own your context window
        messages = [{"role": "system", "content": self._get_system_prompt()}]
        messages.extend(state.get_context_messages())

        # Call LLM
        if not self.config.llm_client:
            raise ValueError("No LLM client configured")

        try:
            response = await self.config.llm_client.chat(
                messages=messages,
                tools=self._get_tools_for_llm(),
            )
        except Exception as e:
            state.add_step(AgentStep(
                step_type=StepType.ERROR,
                content={"error": AgentPrompts.error_context(str(e), "llm_call")},
            ))
            state.status = AgentStatus.FAILED
            return state

        state.current_iteration += 1

        # Process response
        # Factor 1: Natural language to tool calls
        message = response.get("choices", [{}])[0].get("message", {})

        if message.get("tool_calls"):
            # LLM wants to call tools
            for tool_call in message["tool_calls"]:
                state = await self._handle_tool_call(state, tool_call)

                # Check if we're now waiting for human
                if state.status == AgentStatus.WAITING_HUMAN:
                    return state

            # Continue running after tool calls
            state.status = AgentStatus.RUNNING

        elif message.get("content"):
            # LLM provided final response
            state.add_step(AgentStep(
                step_type=StepType.ASSISTANT_MESSAGE,
                content=message["content"],
            ))
            state.status = AgentStatus.COMPLETED

        return state

    async def _handle_tool_call(
        self,
        state: AgentState,
        tool_call: dict[str, Any],
    ) -> AgentState:
        """
        Factor 1: Execute tool calls.
        Factor 7: Handle human contact requests.
        """
        function = tool_call.get("function", {})
        tool_name = function.get("name", "")
        tool_call_id = tool_call.get("id", "")

        try:
            arguments = json.loads(function.get("arguments", "{}"))
        except json.JSONDecodeError:
            arguments = {}

        # Record the tool call
        state.add_step(AgentStep(
            step_type=StepType.TOOL_CALL,
            content={
                "id": tool_call_id,
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": json.dumps(arguments),
                },
            },
        ))

        # Factor 7: Human contact handling
        if tool_name == "request_human_input":
            state.add_step(AgentStep(
                step_type=StepType.HUMAN_REQUEST,
                content={
                    "tool_call_id": tool_call_id,
                    "request_type": arguments.get("request_type"),
                    "message": arguments.get("message"),
                },
            ))
            state.status = AgentStatus.WAITING_HUMAN
            return state

        # Check if tool requires approval
        tool_def = next((t for t in self.config.tools if t.name == tool_name), None)
        if tool_def and tool_def.requires_human_approval:
            state.add_step(AgentStep(
                step_type=StepType.HUMAN_REQUEST,
                content={
                    "tool_call_id": tool_call_id,
                    "request_type": "approval",
                    "message": f"Approve execution of {tool_name} with args: {arguments}",
                },
            ))
            state.status = AgentStatus.WAITING_HUMAN
            return state

        # Execute the tool
        handler = self._tool_handlers.get(tool_name)
        if not handler:
            state.add_step(AgentStep(
                step_type=StepType.ERROR,
                content={
                    "tool_call_id": tool_call_id,
                    "error": f"Unknown tool: {tool_name}",
                },
            ))
            return state

        try:
            result = await handler(**arguments) if callable(handler) else handler
            state.add_step(AgentStep(
                step_type=StepType.TOOL_RESULT,
                content={
                    "tool_call_id": tool_call_id,
                    "result": result,
                },
            ))
        except Exception as e:
            # Factor 9: Compact errors
            state.add_step(AgentStep(
                step_type=StepType.ERROR,
                content={
                    "tool_call_id": tool_call_id,
                    "error": AgentPrompts.error_context(str(e), tool_name),
                },
            ))

        return state

    # Factor 6: Launch/Pause/Resume APIs

    def launch(self, user_message: str) -> AgentState:
        """
        Factor 6: Launch a new agent execution.
        Factor 11: Entry point for triggering agent.
        """
        state = AgentState(
            agent_id=self.config.agent_id,
            status=AgentStatus.RUNNING,
            max_iterations=self.config.max_iterations,
        )
        state.add_step(AgentStep(
            step_type=StepType.USER_MESSAGE,
            content=user_message,
        ))
        return state

    def pause(self, state: AgentState) -> AgentState:
        """Factor 6: Pause execution."""
        if state.status == AgentStatus.RUNNING:
            state.status = AgentStatus.PAUSED
        return state

    def resume(self, state: AgentState) -> AgentState:
        """Factor 6: Resume execution."""
        if state.status == AgentStatus.PAUSED:
            state.status = AgentStatus.RUNNING
        return state

    def provide_human_response(
        self,
        state: AgentState,
        response: str,
        approved: bool = True,
    ) -> AgentState:
        """
        Factor 7: Handle human response.
        """
        if state.status != AgentStatus.WAITING_HUMAN:
            return state

        # Find the pending human request
        for step in reversed(state.steps):
            if step.step_type == StepType.HUMAN_REQUEST:
                tool_call_id = step.content.get("tool_call_id")
                break
        else:
            return state

        state.add_step(AgentStep(
            step_type=StepType.HUMAN_RESPONSE,
            content={
                "tool_call_id": tool_call_id,
                "response": response,
                "approved": approved,
            },
        ))

        # Add as tool result so LLM sees it
        state.add_step(AgentStep(
            step_type=StepType.TOOL_RESULT,
            content={
                "tool_call_id": tool_call_id,
                "result": {
                    "human_response": response,
                    "approved": approved,
                },
            },
        ))

        state.status = AgentStatus.RUNNING
        return state

    async def run_to_completion(
        self,
        state: AgentState,
        human_callback: Callable[[HumanContactRequest], str] | None = None,
    ) -> AgentState:
        """
        Run agent until completion, handling human interactions.
        Factor 8: Own your control flow.
        """
        while state.status == AgentStatus.RUNNING:
            state = await self.step(state)

            if state.status == AgentStatus.WAITING_HUMAN:
                if human_callback:
                    # Find pending request
                    for step in reversed(state.steps):
                        if step.step_type == StepType.HUMAN_REQUEST:
                            request = HumanContactRequest(
                                request_id=step.content.get("tool_call_id", ""),
                                request_type=step.content.get("request_type", ""),
                                message=step.content.get("message", ""),
                            )
                            response = human_callback(request)
                            state = self.provide_human_response(state, response)
                            break
                else:
                    # No callback, stay paused
                    break

        return state
