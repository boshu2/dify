"""
Tests for 12-Factor Agent Implementation.

Tests all 12 factors from https://github.com/humanlayer/12-factor-agents:
1. Natural Language to Tool Calls
2. Own Your Prompts
3. Own Your Context Window
4. Tools are Structured Outputs
5. Unify Execution & Business State
6. Launch/Pause/Resume
7. Contact Humans with Tool Calls
8. Own Your Control Flow
9. Compact Errors into Context
10. Small, Focused Agents
11. Trigger from Anywhere
12. Stateless Reducer
"""
import json
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock

from app.agents.twelve_factor_agent import (
    Agent,
    AgentConfig,
    AgentState,
    AgentStatus,
    AgentStep,
    AgentPrompts,
    HumanContactRequest,
    LLMClient,
    StepType,
    ToolDefinition,
)


class MockLLMClient(LLMClient):
    """Mock LLM client for testing."""

    def __init__(self, responses: list[dict] | None = None):
        self.responses = responses or []
        self.call_count = 0
        self.calls: list[dict] = []

    async def chat(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        self.calls.append({"messages": messages, "tools": tools})
        if self.call_count < len(self.responses):
            response = self.responses[self.call_count]
            self.call_count += 1
            return response
        return {"choices": [{"message": {"content": "Done"}}]}


# =============================================================================
# Factor 1: Natural Language to Tool Calls
# =============================================================================

class TestFactor1NaturalLanguageToToolCalls:
    """Factor 1: LLM converts natural language intent to structured tool calls."""

    @pytest.mark.asyncio
    async def test_llm_converts_intent_to_tool_call(self):
        """LLM should convert user message to appropriate tool calls."""
        # LLM responds with a tool call
        mock_response = {
            "choices": [{
                "message": {
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "search_files",
                            "arguments": '{"query": "test files"}',
                        },
                    }],
                },
            }],
        }

        async def mock_handler(query: str) -> dict:
            return {"files": ["test1.py", "test2.py"]}

        tool = ToolDefinition(
            name="search_files",
            description="Search for files",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
            handler=mock_handler,
        )

        config = AgentConfig(
            agent_id="test-agent",
            purpose="Search files",
            tools=[tool],
            llm_client=MockLLMClient([mock_response]),
        )

        agent = Agent(config)
        state = agent.launch("Find all test files")
        state = await agent.step(state)

        # Verify tool call was made
        tool_call_steps = [s for s in state.steps if s.step_type == StepType.TOOL_CALL]
        assert len(tool_call_steps) == 1
        assert tool_call_steps[0].content["function"]["name"] == "search_files"

    @pytest.mark.asyncio
    async def test_tool_execution_returns_structured_result(self):
        """Tool results should be structured and returned to LLM."""
        mock_responses = [
            # First: LLM calls tool
            {
                "choices": [{
                    "message": {
                        "tool_calls": [{
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city": "NYC"}',
                            },
                        }],
                    },
                }],
            },
            # Second: LLM provides final response
            {"choices": [{"message": {"content": "The weather is sunny."}}]},
        ]

        async def get_weather(city: str) -> dict:
            return {"city": city, "temp": 72, "condition": "sunny"}

        tool = ToolDefinition(
            name="get_weather",
            description="Get weather for a city",
            parameters={
                "type": "object",
                "properties": {"city": {"type": "string"}},
            },
            handler=get_weather,
        )

        config = AgentConfig(
            agent_id="weather-agent",
            purpose="Get weather",
            tools=[tool],
            llm_client=MockLLMClient(mock_responses),
        )

        agent = Agent(config)
        state = agent.launch("What's the weather in NYC?")
        state = await agent.run_to_completion(state)

        # Check tool result was recorded
        result_steps = [s for s in state.steps if s.step_type == StepType.TOOL_RESULT]
        assert len(result_steps) == 1
        assert result_steps[0].content["result"]["city"] == "NYC"


# =============================================================================
# Factor 2: Own Your Prompts
# =============================================================================

class TestFactor2OwnYourPrompts:
    """Factor 2: Full control over prompt templates."""

    def test_system_prompt_includes_purpose(self):
        """System prompt should include agent purpose."""
        tools = [
            ToolDefinition(
                name="search",
                description="Search the web",
                parameters={"type": "object"},
            ),
        ]
        prompt = AgentPrompts.system_prompt("Web search assistant", tools)

        assert "Web search assistant" in prompt
        assert "search: Search the web" in prompt

    def test_system_prompt_lists_all_tools(self):
        """System prompt should list all available tools."""
        tools = [
            ToolDefinition(name="tool_a", description="Does A", parameters={}),
            ToolDefinition(name="tool_b", description="Does B", parameters={}),
            ToolDefinition(name="tool_c", description="Does C", parameters={}),
        ]
        prompt = AgentPrompts.system_prompt("Multi-tool agent", tools)

        assert "tool_a: Does A" in prompt
        assert "tool_b: Does B" in prompt
        assert "tool_c: Does C" in prompt

    def test_error_context_is_compact(self):
        """Error context should be truncated for context efficiency."""
        long_error = "x" * 500
        compact = AgentPrompts.error_context(long_error, "failing_tool")

        assert len(compact) < 300  # Should truncate
        assert "failing_tool" in compact


# =============================================================================
# Factor 3: Own Your Context Window
# =============================================================================

class TestFactor3OwnYourContextWindow:
    """Factor 3: Deliberate context window construction."""

    def test_context_includes_user_messages(self):
        """Context should include user messages."""
        state = AgentState(agent_id="test")
        state.add_step(AgentStep(step_type=StepType.USER_MESSAGE, content="Hello"))

        messages = state.get_context_messages()
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello"

    def test_context_includes_assistant_messages(self):
        """Context should include assistant messages."""
        state = AgentState(agent_id="test")
        state.add_step(AgentStep(step_type=StepType.ASSISTANT_MESSAGE, content="Hi there"))

        messages = state.get_context_messages()
        assert len(messages) == 1
        assert messages[0]["role"] == "assistant"

    def test_context_includes_tool_calls(self):
        """Context should include tool calls in LLM format."""
        state = AgentState(agent_id="test")
        state.add_step(AgentStep(
            step_type=StepType.TOOL_CALL,
            content={
                "id": "call_1",
                "type": "function",
                "function": {"name": "search", "arguments": "{}"},
            },
        ))

        messages = state.get_context_messages()
        assert messages[0]["role"] == "assistant"
        assert messages[0]["tool_calls"][0]["id"] == "call_1"

    def test_context_includes_tool_results(self):
        """Context should include tool results."""
        state = AgentState(agent_id="test")
        state.add_step(AgentStep(
            step_type=StepType.TOOL_RESULT,
            content={"tool_call_id": "call_1", "result": {"data": "value"}},
        ))

        messages = state.get_context_messages()
        assert messages[0]["role"] == "tool"
        assert messages[0]["tool_call_id"] == "call_1"

    def test_context_preserves_message_order(self):
        """Context should preserve chronological message order."""
        state = AgentState(agent_id="test")
        state.add_step(AgentStep(step_type=StepType.USER_MESSAGE, content="1"))
        state.add_step(AgentStep(step_type=StepType.ASSISTANT_MESSAGE, content="2"))
        state.add_step(AgentStep(step_type=StepType.USER_MESSAGE, content="3"))

        messages = state.get_context_messages()
        assert len(messages) == 3
        assert messages[0]["content"] == "1"
        assert messages[1]["content"] == "2"
        assert messages[2]["content"] == "3"


# =============================================================================
# Factor 4: Tools are Structured Outputs
# =============================================================================

class TestFactor4ToolsAreStructuredOutputs:
    """Factor 4: Tool calls as constrained output format."""

    def test_tool_converts_to_openai_format(self):
        """Tool should convert to OpenAI function format."""
        tool = ToolDefinition(
            name="calculate",
            description="Perform calculation",
            parameters={
                "type": "object",
                "properties": {
                    "expression": {"type": "string"},
                },
                "required": ["expression"],
            },
        )

        openai_format = tool.to_openai_format()

        assert openai_format["type"] == "function"
        assert openai_format["function"]["name"] == "calculate"
        assert openai_format["function"]["description"] == "Perform calculation"
        assert "expression" in openai_format["function"]["parameters"]["properties"]

    def test_agent_provides_tools_to_llm(self):
        """Agent should provide tool definitions to LLM."""
        tool = ToolDefinition(
            name="test_tool",
            description="Test",
            parameters={"type": "object"},
        )

        mock_client = MockLLMClient([
            {"choices": [{"message": {"content": "Done"}}]},
        ])

        config = AgentConfig(
            agent_id="test",
            purpose="Test",
            tools=[tool],
            llm_client=mock_client,
        )

        agent = Agent(config)
        state = agent.launch("Test")

        import asyncio
        asyncio.get_event_loop().run_until_complete(agent.step(state))

        # Check that tools were passed to LLM
        assert len(mock_client.calls) == 1
        tools_sent = mock_client.calls[0]["tools"]
        # Should have test_tool + request_human_input
        assert len(tools_sent) >= 1
        tool_names = [t["function"]["name"] for t in tools_sent]
        assert "test_tool" in tool_names


# =============================================================================
# Factor 5: Unify Execution & Business State
# =============================================================================

class TestFactor5UnifyExecutionAndBusinessState:
    """Factor 5: State inferable from execution history."""

    def test_state_captures_all_steps(self):
        """AgentState should capture complete execution history."""
        state = AgentState(agent_id="test")

        state.add_step(AgentStep(step_type=StepType.USER_MESSAGE, content="Q1"))
        state.add_step(AgentStep(step_type=StepType.ASSISTANT_MESSAGE, content="A1"))
        state.add_step(AgentStep(step_type=StepType.TOOL_CALL, content={"name": "search"}))
        state.add_step(AgentStep(step_type=StepType.TOOL_RESULT, content={"result": "data"}))

        assert len(state.steps) == 4

    def test_state_is_serializable(self):
        """State should be serializable to dict."""
        state = AgentState(
            agent_id="test",
            status=AgentStatus.RUNNING,
            current_iteration=2,
        )
        state.add_step(AgentStep(step_type=StepType.USER_MESSAGE, content="Hello"))

        data = state.to_dict()

        assert data["agent_id"] == "test"
        assert data["status"] == "running"
        assert data["current_iteration"] == 2
        assert len(data["steps"]) == 1

    def test_state_is_deserializable(self):
        """State should be restorable from dict."""
        original = AgentState(
            agent_id="test",
            status=AgentStatus.PAUSED,
            current_iteration=5,
        )
        original.add_step(AgentStep(step_type=StepType.USER_MESSAGE, content="Test"))

        data = original.to_dict()
        restored = AgentState.from_dict(data)

        assert restored.agent_id == "test"
        assert restored.status == AgentStatus.PAUSED
        assert restored.current_iteration == 5
        assert len(restored.steps) == 1
        assert restored.steps[0].content == "Test"

    def test_state_hash_changes_on_modification(self):
        """State hash should change when state changes."""
        state = AgentState(agent_id="test")
        hash1 = state.compute_hash()

        state.add_step(AgentStep(step_type=StepType.USER_MESSAGE, content="New"))
        hash2 = state.compute_hash()

        assert hash1 != hash2


# =============================================================================
# Factor 6: Launch/Pause/Resume
# =============================================================================

class TestFactor6LaunchPauseResume:
    """Factor 6: Simple lifecycle management APIs."""

    def test_launch_creates_running_state(self):
        """Launch should create a running agent state."""
        config = AgentConfig(agent_id="test", purpose="Test")
        agent = Agent(config)

        state = agent.launch("Start task")

        assert state.status == AgentStatus.RUNNING
        assert state.agent_id == "test"
        assert len(state.steps) == 1
        assert state.steps[0].step_type == StepType.USER_MESSAGE

    def test_pause_sets_paused_status(self):
        """Pause should set state to paused."""
        config = AgentConfig(agent_id="test", purpose="Test")
        agent = Agent(config)

        state = agent.launch("Start")
        state = agent.pause(state)

        assert state.status == AgentStatus.PAUSED

    def test_resume_sets_running_status(self):
        """Resume should set paused state to running."""
        config = AgentConfig(agent_id="test", purpose="Test")
        agent = Agent(config)

        state = agent.launch("Start")
        state = agent.pause(state)
        state = agent.resume(state)

        assert state.status == AgentStatus.RUNNING

    def test_pause_only_affects_running(self):
        """Pause should only affect running state."""
        config = AgentConfig(agent_id="test", purpose="Test")
        agent = Agent(config)

        state = AgentState(agent_id="test", status=AgentStatus.COMPLETED)
        state = agent.pause(state)

        assert state.status == AgentStatus.COMPLETED  # Unchanged

    def test_resume_only_affects_paused(self):
        """Resume should only affect paused state."""
        config = AgentConfig(agent_id="test", purpose="Test")
        agent = Agent(config)

        state = AgentState(agent_id="test", status=AgentStatus.COMPLETED)
        state = agent.resume(state)

        assert state.status == AgentStatus.COMPLETED  # Unchanged


# =============================================================================
# Factor 7: Contact Humans with Tool Calls
# =============================================================================

class TestFactor7ContactHumansWithToolCalls:
    """Factor 7: Human-in-the-loop via tool calls."""

    def test_agent_has_human_contact_tool(self):
        """Agent should automatically have human contact tool."""
        config = AgentConfig(agent_id="test", purpose="Test")
        agent = Agent(config)

        tool_names = [t.name for t in agent.config.tools]
        assert "request_human_input" in tool_names

    @pytest.mark.asyncio
    async def test_human_request_pauses_agent(self):
        """Human request should pause agent execution."""
        mock_response = {
            "choices": [{
                "message": {
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "request_human_input",
                            "arguments": json.dumps({
                                "request_type": "approval",
                                "message": "Should I proceed?",
                            }),
                        },
                    }],
                },
            }],
        }

        config = AgentConfig(
            agent_id="test",
            purpose="Test",
            llm_client=MockLLMClient([mock_response]),
        )

        agent = Agent(config)
        state = agent.launch("Do something that needs approval")
        state = await agent.step(state)

        assert state.status == AgentStatus.WAITING_HUMAN

    @pytest.mark.asyncio
    async def test_human_response_resumes_agent(self):
        """Providing human response should resume agent."""
        mock_responses = [
            # First: LLM requests human input
            {
                "choices": [{
                    "message": {
                        "tool_calls": [{
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "request_human_input",
                                "arguments": json.dumps({
                                    "request_type": "input",
                                    "message": "What is your name?",
                                }),
                            },
                        }],
                    },
                }],
            },
            # After human response: complete
            {"choices": [{"message": {"content": "Hello, John!"}}]},
        ]

        config = AgentConfig(
            agent_id="test",
            purpose="Greet user",
            llm_client=MockLLMClient(mock_responses),
        )

        agent = Agent(config)
        state = agent.launch("Greet me")
        state = await agent.step(state)

        assert state.status == AgentStatus.WAITING_HUMAN

        state = agent.provide_human_response(state, "John")

        assert state.status == AgentStatus.RUNNING

        # Human response should be in steps
        human_response_steps = [s for s in state.steps if s.step_type == StepType.HUMAN_RESPONSE]
        assert len(human_response_steps) == 1
        assert human_response_steps[0].content["response"] == "John"

    @pytest.mark.asyncio
    async def test_tool_requiring_approval_pauses(self):
        """Tools marked as requiring approval should pause for human."""
        mock_response = {
            "choices": [{
                "message": {
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "delete_file",
                            "arguments": '{"path": "/important.txt"}',
                        },
                    }],
                },
            }],
        }

        async def delete_file(path: str) -> dict:
            return {"deleted": path}

        tool = ToolDefinition(
            name="delete_file",
            description="Delete a file",
            parameters={"type": "object"},
            handler=delete_file,
            requires_human_approval=True,  # Dangerous operation
        )

        config = AgentConfig(
            agent_id="test",
            purpose="File manager",
            tools=[tool],
            llm_client=MockLLMClient([mock_response]),
        )

        agent = Agent(config)
        state = agent.launch("Delete important.txt")
        state = await agent.step(state)

        assert state.status == AgentStatus.WAITING_HUMAN


# =============================================================================
# Factor 8: Own Your Control Flow
# =============================================================================

class TestFactor8OwnYourControlFlow:
    """Factor 8: Explicit state transitions."""

    @pytest.mark.asyncio
    async def test_agent_stops_on_completion(self):
        """Agent should stop when LLM provides final response."""
        config = AgentConfig(
            agent_id="test",
            purpose="Simple response",
            llm_client=MockLLMClient([
                {"choices": [{"message": {"content": "Here is your answer."}}]},
            ]),
        )

        agent = Agent(config)
        state = agent.launch("Answer me")
        state = await agent.step(state)

        assert state.status == AgentStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_agent_continues_after_tool_call(self):
        """Agent should continue running after tool execution."""
        mock_responses = [
            # Tool call
            {
                "choices": [{
                    "message": {
                        "tool_calls": [{
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "test_tool", "arguments": "{}"},
                        }],
                    },
                }],
            },
        ]

        async def test_handler() -> dict:
            return {"ok": True}

        tool = ToolDefinition(
            name="test_tool",
            description="Test",
            parameters={},
            handler=test_handler,
        )

        config = AgentConfig(
            agent_id="test",
            purpose="Test",
            tools=[tool],
            llm_client=MockLLMClient(mock_responses),
        )

        agent = Agent(config)
        state = agent.launch("Call tool")
        state = await agent.step(state)

        # Should still be running for next step
        assert state.status == AgentStatus.RUNNING

    @pytest.mark.asyncio
    async def test_max_iterations_causes_failure(self):
        """Agent should fail when max iterations exceeded."""
        # LLM always returns a tool call, never completes
        tool_call_response = {
            "choices": [{
                "message": {
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "endless", "arguments": "{}"},
                    }],
                },
            }],
        }

        async def endless_handler() -> dict:
            return {"continue": True}

        tool = ToolDefinition(
            name="endless",
            description="Endless",
            parameters={},
            handler=endless_handler,
        )

        config = AgentConfig(
            agent_id="test",
            purpose="Test",
            tools=[tool],
            max_iterations=3,
            llm_client=MockLLMClient([tool_call_response] * 10),
        )

        agent = Agent(config)
        state = agent.launch("Loop forever")

        # Run multiple steps
        for _ in range(5):
            if state.status == AgentStatus.FAILED:
                break
            state = await agent.step(state)

        assert state.status == AgentStatus.FAILED
        error_steps = [s for s in state.steps if s.step_type == StepType.ERROR]
        assert any("Max iterations" in str(s.content) for s in error_steps)


# =============================================================================
# Factor 9: Compact Errors into Context
# =============================================================================

class TestFactor9CompactErrorsIntoContext:
    """Factor 9: Efficient error representation."""

    def test_errors_appear_in_context(self):
        """Errors should appear in context for LLM to see."""
        state = AgentState(agent_id="test")
        state.add_step(AgentStep(
            step_type=StepType.ERROR,
            content={"tool_call_id": "call_1", "error": "Connection failed"},
        ))

        messages = state.get_context_messages()

        assert len(messages) == 1
        assert messages[0]["role"] == "tool"
        assert "Error:" in messages[0]["content"]
        assert "Connection failed" in messages[0]["content"]

    @pytest.mark.asyncio
    async def test_tool_error_is_recorded(self):
        """Tool execution errors should be recorded as error steps."""
        mock_response = {
            "choices": [{
                "message": {
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "failing_tool", "arguments": "{}"},
                    }],
                },
            }],
        }

        async def failing_handler() -> dict:
            raise ValueError("Something went wrong!")

        tool = ToolDefinition(
            name="failing_tool",
            description="Always fails",
            parameters={},
            handler=failing_handler,
        )

        config = AgentConfig(
            agent_id="test",
            purpose="Test",
            tools=[tool],
            llm_client=MockLLMClient([mock_response]),
        )

        agent = Agent(config)
        state = agent.launch("Call failing tool")
        state = await agent.step(state)

        error_steps = [s for s in state.steps if s.step_type == StepType.ERROR]
        assert len(error_steps) == 1
        assert "Something went wrong" in str(error_steps[0].content)


# =============================================================================
# Factor 10: Small, Focused Agents
# =============================================================================

class TestFactor10SmallFocusedAgents:
    """Factor 10: Single-purpose agents."""

    def test_agent_has_defined_purpose(self):
        """Agent should have a clear, defined purpose."""
        config = AgentConfig(
            agent_id="code-reviewer",
            purpose="Review code for security vulnerabilities",
        )
        agent = Agent(config)

        assert "security vulnerabilities" in agent.config.purpose

    def test_purpose_appears_in_system_prompt(self):
        """Agent purpose should appear in system prompt."""
        config = AgentConfig(
            agent_id="test",
            purpose="Analyze financial data",
        )
        agent = Agent(config)

        prompt = agent._get_system_prompt()
        assert "Analyze financial data" in prompt


# =============================================================================
# Factor 11: Trigger from Anywhere
# =============================================================================

class TestFactor11TriggerFromAnywhere:
    """Factor 11: Multiple entry points."""

    def test_can_trigger_via_launch(self):
        """Agent can be triggered via launch() method."""
        config = AgentConfig(agent_id="test", purpose="Test")
        agent = Agent(config)

        state = agent.launch("Trigger from launch")

        assert state.status == AgentStatus.RUNNING
        assert state.steps[0].content == "Trigger from launch"

    def test_can_resume_from_serialized_state(self):
        """Agent can resume from serialized state (enabling async triggers)."""
        config = AgentConfig(agent_id="test", purpose="Test")
        agent = Agent(config)

        # Create and serialize state
        state = agent.launch("Initial trigger")
        state = agent.pause(state)
        serialized = state.to_dict()

        # "Later" - restore and resume
        restored = AgentState.from_dict(serialized)
        restored = agent.resume(restored)

        assert restored.status == AgentStatus.RUNNING


# =============================================================================
# Factor 12: Stateless Reducer
# =============================================================================

class TestFactor12StatelessReducer:
    """Factor 12: Agent as pure state transformer."""

    @pytest.mark.asyncio
    async def test_step_returns_new_state(self):
        """Step should return modified state, not modify in place conceptually."""
        config = AgentConfig(
            agent_id="test",
            purpose="Test",
            llm_client=MockLLMClient([
                {"choices": [{"message": {"content": "Response"}}]},
            ]),
        )

        agent = Agent(config)
        initial_state = agent.launch("Test")
        initial_hash = initial_state.compute_hash()

        new_state = await agent.step(initial_state)
        new_hash = new_state.compute_hash()

        # State changed (new step added)
        assert new_hash != initial_hash

    @pytest.mark.asyncio
    async def test_agent_holds_no_execution_state(self):
        """Agent should not hold execution state between calls."""
        config = AgentConfig(
            agent_id="test",
            purpose="Test",
            llm_client=MockLLMClient([
                {"choices": [{"message": {"content": "R1"}}]},
                {"choices": [{"message": {"content": "R2"}}]},
            ]),
        )

        agent = Agent(config)

        # Two independent executions
        state1 = agent.launch("Task 1")
        state1 = await agent.step(state1)

        state2 = agent.launch("Task 2")
        state2 = await agent.step(state2)

        # States are independent
        assert state1.agent_id == state2.agent_id == "test"
        assert state1.steps[0].content == "Task 1"
        assert state2.steps[0].content == "Task 2"

    def test_same_input_produces_deterministic_context(self):
        """Same state should produce same context messages."""
        state = AgentState(agent_id="test")
        state.add_step(AgentStep(step_type=StepType.USER_MESSAGE, content="Hello"))
        state.add_step(AgentStep(step_type=StepType.ASSISTANT_MESSAGE, content="Hi"))

        context1 = state.get_context_messages()
        context2 = state.get_context_messages()

        assert context1 == context2


# =============================================================================
# Integration Tests
# =============================================================================

class TestAgentIntegration:
    """Integration tests for the 12-factor agent."""

    @pytest.mark.asyncio
    async def test_full_conversation_flow(self):
        """Test a complete agent conversation."""
        mock_responses = [
            # First: tool call
            {
                "choices": [{
                    "message": {
                        "tool_calls": [{
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "get_data",
                                "arguments": '{"key": "user_info"}',
                            },
                        }],
                    },
                }],
            },
            # Second: final response
            {"choices": [{"message": {"content": "Based on the data, your name is Alice."}}]},
        ]

        async def get_data(key: str) -> dict:
            return {"user_info": {"name": "Alice", "role": "admin"}}

        tool = ToolDefinition(
            name="get_data",
            description="Get data by key",
            parameters={
                "type": "object",
                "properties": {"key": {"type": "string"}},
            },
            handler=get_data,
        )

        config = AgentConfig(
            agent_id="data-agent",
            purpose="Retrieve and analyze data",
            tools=[tool],
            llm_client=MockLLMClient(mock_responses),
        )

        agent = Agent(config)
        state = agent.launch("What is my name?")
        state = await agent.run_to_completion(state)

        assert state.status == AgentStatus.COMPLETED

        # Check conversation flow
        step_types = [s.step_type for s in state.steps]
        assert StepType.USER_MESSAGE in step_types
        assert StepType.TOOL_CALL in step_types
        assert StepType.TOOL_RESULT in step_types
        assert StepType.ASSISTANT_MESSAGE in step_types

    @pytest.mark.asyncio
    async def test_human_in_the_loop_flow(self):
        """Test human-in-the-loop conversation."""
        mock_responses = [
            # First: request human approval
            {
                "choices": [{
                    "message": {
                        "tool_calls": [{
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "request_human_input",
                                "arguments": json.dumps({
                                    "request_type": "approval",
                                    "message": "Proceed with deletion?",
                                }),
                            },
                        }],
                    },
                }],
            },
            # After approval: complete
            {"choices": [{"message": {"content": "Deletion approved and completed."}}]},
        ]

        config = AgentConfig(
            agent_id="test",
            purpose="Test with human approval",
            llm_client=MockLLMClient(mock_responses),
        )

        agent = Agent(config)

        def human_callback(request: HumanContactRequest) -> str:
            return "Yes, proceed"

        state = agent.launch("Delete the files")
        state = await agent.run_to_completion(state, human_callback=human_callback)

        assert state.status == AgentStatus.COMPLETED

        # Verify human interaction happened
        human_request_steps = [s for s in state.steps if s.step_type == StepType.HUMAN_REQUEST]
        human_response_steps = [s for s in state.steps if s.step_type == StepType.HUMAN_RESPONSE]

        assert len(human_request_steps) == 1
        assert len(human_response_steps) == 1
