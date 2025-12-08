"""
Integration tests for 12-factor agent workflows.

Tests end-to-end agent scenarios including:
- Human-in-the-loop interactions
- RAG-enhanced conversations
- Workflow execution with agents
- State persistence and resumption
"""
import pytest
import json

from app.agents.twelve_factor_agent import (
    Agent,
    AgentConfig,
    AgentState,
    AgentStatus,
    LLMClient,
    StepType,
)
from app.agents.rag_agent import RAGAgent, RAGAgentConfig
from app.rag.retriever import BM25Retriever
from app.rag.vector_store import Document
from app.workflows import (
    WorkflowBuilder,
    WorkflowEngine,
    WorkflowState,
    WorkflowStatus,
)


class MockLLMClient(LLMClient):
    """Mock LLM client for integration tests."""

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
        return {"choices": [{"message": {"content": "Default response"}}]}


class TestHumanInTheLoopIntegration:
    """Integration tests for human-in-the-loop workflows."""

    @pytest.mark.asyncio
    async def test_agent_requests_human_approval(self):
        """Agent should pause when requesting human approval."""
        # LLM decides to request human input
        mock_responses = [
            {
                "choices": [{
                    "message": {
                        "tool_calls": [{
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "request_human_input",
                                "arguments": json.dumps({
                                    "question": "Should I proceed with deleting the file?",
                                    "context": "User requested file deletion",
                                }),
                            },
                        }],
                    },
                }],
            },
        ]

        agent = Agent(AgentConfig(
            agent_id="approval-agent",
            purpose="Handle file operations with approval",
            tools=[],
            llm_client=MockLLMClient(mock_responses),
        ))

        state = agent.launch("Delete the config.yaml file")
        state = await agent.step(state)

        # Agent should be waiting for human input
        assert state.status == AgentStatus.WAITING_HUMAN

    @pytest.mark.asyncio
    async def test_agent_continues_after_human_approval(self):
        """Agent should continue after receiving human approval."""
        mock_responses = [
            # First: request approval
            {
                "choices": [{
                    "message": {
                        "tool_calls": [{
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "request_human_input",
                                "arguments": json.dumps({
                                    "question": "Proceed with deployment?",
                                }),
                            },
                        }],
                    },
                }],
            },
            # After approval: complete
            {
                "choices": [{
                    "message": {
                        "content": "Deployment approved and completed successfully.",
                    },
                }],
            },
        ]

        agent = Agent(AgentConfig(
            agent_id="deploy-agent",
            purpose="Handle deployments",
            llm_client=MockLLMClient(mock_responses),
        ))

        # Start and get to human input request
        state = agent.launch("Deploy to production")
        state = await agent.step(state)
        assert state.status == AgentStatus.WAITING_HUMAN

        # Provide human input
        state = agent.provide_human_response(state, "Yes, proceed with deployment")
        assert state.status == AgentStatus.RUNNING

        # Continue to completion
        state = await agent.run_to_completion(state)
        assert state.status == AgentStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_agent_handles_human_rejection(self):
        """Agent should handle human rejection gracefully."""
        mock_responses = [
            # Request approval
            {
                "choices": [{
                    "message": {
                        "tool_calls": [{
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "request_human_input",
                                "arguments": json.dumps({
                                    "question": "Delete all user data?",
                                }),
                            },
                        }],
                    },
                }],
            },
            # After rejection: acknowledge
            {
                "choices": [{
                    "message": {
                        "content": "Understood. Operation cancelled as per your request.",
                    },
                }],
            },
        ]

        agent = Agent(AgentConfig(
            agent_id="cautious-agent",
            purpose="Handle destructive operations carefully",
            llm_client=MockLLMClient(mock_responses),
        ))

        state = agent.launch("Delete all user data")
        state = await agent.step(state)

        # Provide rejection (approved=False)
        state = agent.provide_human_response(state, "No, do not delete anything", approved=False)
        state = await agent.run_to_completion(state)

        assert state.status == AgentStatus.COMPLETED
        # Verify the rejection was passed to LLM
        last_call = agent.config.llm_client.calls[-1]
        messages = last_call["messages"]
        tool_result_msgs = [m for m in messages if m.get("role") == "tool"]
        assert any("No, do not delete" in str(m.get("content", "")) for m in tool_result_msgs)


class TestRAGAgentIntegration:
    """Integration tests for RAG-enhanced agents."""

    @pytest.mark.asyncio
    async def test_rag_agent_uses_knowledge_base(self):
        """RAG agent should retrieve and use knowledge."""
        # Set up knowledge base
        docs = [
            Document(id="doc1", content="The API rate limit is 100 requests per minute."),
            Document(id="doc2", content="Authentication requires an API key in the header."),
            Document(id="doc3", content="All responses are in JSON format."),
        ]
        retriever = BM25Retriever(documents=docs)

        mock_responses = [
            # Agent searches knowledge
            {
                "choices": [{
                    "message": {
                        "tool_calls": [{
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "knowledge_search",
                                "arguments": json.dumps({"query": "rate limit"}),
                            },
                        }],
                    },
                }],
            },
            # Agent responds with knowledge
            {
                "choices": [{
                    "message": {
                        "content": "Based on my knowledge base, the API rate limit is 100 requests per minute.",
                    },
                }],
            },
        ]

        agent = RAGAgent(RAGAgentConfig(
            agent_id="api-assistant",
            purpose="Answer questions about our API",
            retriever=retriever,
            llm_client=MockLLMClient(mock_responses),
        ))

        state = agent.launch("What is the rate limit?")
        state = await agent.run_to_completion(state)

        assert state.status == AgentStatus.COMPLETED

        # Verify knowledge_search was called
        step_types = [s.step_type for s in state.steps]
        assert StepType.TOOL_CALL in step_types
        assert StepType.TOOL_RESULT in step_types

    @pytest.mark.asyncio
    async def test_rag_agent_handles_no_results(self):
        """RAG agent should handle queries with no matching documents."""
        # Empty knowledge base
        retriever = BM25Retriever(documents=[])

        mock_responses = [
            # Agent searches
            {
                "choices": [{
                    "message": {
                        "tool_calls": [{
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "knowledge_search",
                                "arguments": json.dumps({"query": "quantum physics"}),
                            },
                        }],
                    },
                }],
            },
            # Agent acknowledges no results
            {
                "choices": [{
                    "message": {
                        "content": "I don't have information about quantum physics in my knowledge base.",
                    },
                }],
            },
        ]

        agent = RAGAgent(RAGAgentConfig(
            agent_id="limited-agent",
            purpose="Answer questions",
            retriever=retriever,
            llm_client=MockLLMClient(mock_responses),
        ))

        state = agent.launch("Explain quantum entanglement")
        state = await agent.run_to_completion(state)

        assert state.status == AgentStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_rag_agent_multiple_searches(self):
        """RAG agent should support multiple knowledge searches."""
        docs = [
            Document(id="pricing", content="Basic plan costs $10/month."),
            Document(id="features", content="Basic plan includes 1000 API calls."),
        ]
        retriever = BM25Retriever(documents=docs)

        mock_responses = [
            # First search: pricing
            {
                "choices": [{
                    "message": {
                        "tool_calls": [{
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "knowledge_search",
                                "arguments": json.dumps({"query": "pricing cost"}),
                            },
                        }],
                    },
                }],
            },
            # Second search: features
            {
                "choices": [{
                    "message": {
                        "tool_calls": [{
                            "id": "call_2",
                            "type": "function",
                            "function": {
                                "name": "knowledge_search",
                                "arguments": json.dumps({"query": "features included"}),
                            },
                        }],
                    },
                }],
            },
            # Final response
            {
                "choices": [{
                    "message": {
                        "content": "The Basic plan costs $10/month and includes 1000 API calls.",
                    },
                }],
            },
        ]

        agent = RAGAgent(RAGAgentConfig(
            agent_id="sales-agent",
            purpose="Answer pricing and feature questions",
            retriever=retriever,
            llm_client=MockLLMClient(mock_responses),
        ))

        state = agent.launch("What do I get with the Basic plan and how much does it cost?")
        state = await agent.run_to_completion(state)

        assert state.status == AgentStatus.COMPLETED

        # Verify multiple tool calls
        tool_calls = [s for s in state.steps if s.step_type == StepType.TOOL_CALL]
        assert len(tool_calls) >= 2


class TestWorkflowAgentIntegration:
    """Integration tests for workflows with agents."""

    @pytest.mark.asyncio
    async def test_simple_workflow_execution(self):
        """Test simple workflow execution."""
        workflow = (
            WorkflowBuilder("test-workflow", "Test Workflow")
            .add_start("start")
            .add_transform("process", "passthrough")
            .add_end("end")
            .connect("start", "process")
            .connect("process", "end")
            .build()
        )

        engine = WorkflowEngine()
        state = engine.launch(workflow)
        state = await engine.start(workflow, state, input_data={"input": "test"})
        state = await engine.run_to_completion(workflow, state)

        assert state.status == WorkflowStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_workflow_with_human_node(self):
        """Workflow should pause at human nodes."""
        workflow = (
            WorkflowBuilder("approval-workflow", "Approval Workflow")
            .add_start("start")
            .add_human("approve", "Please approve this action")
            .add_end("end")
            .connect("start", "approve")
            .connect("approve", "end")
            .build()
        )

        engine = WorkflowEngine()
        state = engine.launch(workflow)
        state = await engine.start(workflow, state, input_data={"request": "deploy"})
        state = await engine.run_to_completion(workflow, state)

        # Should pause at human node
        assert state.status == WorkflowStatus.WAITING_HUMAN

        # Provide approval (node_id is "approve")
        state = await engine.provide_human_response(workflow, state, "approve", "Approved")
        state = await engine.run_to_completion(workflow, state)

        assert state.status == WorkflowStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_workflow_with_condition(self):
        """Workflow should route based on conditions."""
        workflow = (
            WorkflowBuilder("conditional-workflow", "Conditional Workflow")
            .add_start("start")
            .add_condition("check", "variables.score > 50")
            .add_transform("high", "passthrough")
            .add_transform("low", "passthrough")
            .add_end("end")
            .connect("start", "check")
            .connect("check", "high", condition="true")
            .connect("check", "low", condition="false")
            .connect("high", "end")
            .connect("low", "end")
            .build()
        )

        engine = WorkflowEngine()

        # Test with high score
        state = engine.launch(workflow)
        state = await engine.start(workflow, state, input_data={"score": 75})
        state = await engine.run_to_completion(workflow, state)
        assert state.status == WorkflowStatus.COMPLETED


class TestStatePersistenceIntegration:
    """Integration tests for state persistence and resumption."""

    @pytest.mark.asyncio
    async def test_agent_state_serialization_roundtrip(self):
        """Agent state should survive serialization/deserialization."""
        mock_responses = [
            {
                "choices": [{
                    "message": {
                        "tool_calls": [{
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "request_human_input",
                                "arguments": json.dumps({"question": "Continue?"}),
                            },
                        }],
                    },
                }],
            },
            {
                "choices": [{
                    "message": {"content": "Continued after pause."},
                }],
            },
        ]

        # Create agent and run to pause
        agent1 = Agent(AgentConfig(
            agent_id="pausable-agent",
            purpose="Demo pause/resume",
            llm_client=MockLLMClient(mock_responses[:1]),
        ))

        state1 = agent1.launch("Do something")
        state1 = await agent1.step(state1)
        assert state1.status == AgentStatus.WAITING_HUMAN

        # Serialize state
        serialized = state1.to_dict()
        serialized_json = json.dumps(serialized)

        # Deserialize and resume with new agent instance
        deserialized = json.loads(serialized_json)
        state2 = AgentState.from_dict(deserialized)

        agent2 = Agent(AgentConfig(
            agent_id="pausable-agent",
            purpose="Demo pause/resume",
            llm_client=MockLLMClient(mock_responses[1:]),
        ))

        # Provide input and continue
        state2 = agent2.provide_human_response(state2, "Yes")
        state2 = await agent2.run_to_completion(state2)

        assert state2.status == AgentStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_workflow_state_persistence(self):
        """Workflow state should persist across restarts."""
        workflow = (
            WorkflowBuilder("persist-workflow", "Persist Workflow")
            .add_start("start")
            .add_human("check", "Review before proceeding")
            .add_transform("increment", "passthrough")
            .add_end("end")
            .connect("start", "check")
            .connect("check", "increment")
            .connect("increment", "end")
            .build()
        )

        engine1 = WorkflowEngine()
        state1 = engine1.launch(workflow)
        state1 = await engine1.start(workflow, state1, input_data={"count": 0})
        state1 = await engine1.run_to_completion(workflow, state1)

        # Should pause at human node
        assert state1.status == WorkflowStatus.WAITING_HUMAN

        # Serialize
        serialized = state1.to_dict()

        # Restore in new engine
        engine2 = WorkflowEngine()
        state2 = WorkflowState.from_dict(serialized)

        # Resume (node_id is "check")
        state2 = await engine2.provide_human_response(workflow, state2, "check", "OK")
        state2 = await engine2.run_to_completion(workflow, state2)

        assert state2.status == WorkflowStatus.COMPLETED


class TestEndToEndScenarios:
    """End-to-end integration tests for realistic scenarios."""

    @pytest.mark.asyncio
    async def test_customer_support_scenario(self):
        """Test a customer support agent with RAG."""
        # Knowledge base with support articles
        docs = [
            Document(
                id="refund",
                content="Refunds are processed within 5-7 business days.",
                metadata={"category": "billing"},
            ),
            Document(
                id="password",
                content="To reset your password, click 'Forgot Password' on the login page.",
                metadata={"category": "account"},
            ),
        ]
        retriever = BM25Retriever(documents=docs)

        mock_responses = [
            # Search knowledge
            {
                "choices": [{
                    "message": {
                        "tool_calls": [{
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "knowledge_search",
                                "arguments": json.dumps({"query": "refund processing time"}),
                            },
                        }],
                    },
                }],
            },
            # Respond with knowledge
            {
                "choices": [{
                    "message": {
                        "content": "Refunds are typically processed within 5-7 business days.",
                    },
                }],
            },
        ]

        agent = RAGAgent(RAGAgentConfig(
            agent_id="support-bot",
            purpose="Help customers with questions",
            retriever=retriever,
            llm_client=MockLLMClient(mock_responses),
        ))

        state = agent.launch("How long does a refund take?")
        state = await agent.run_to_completion(state)

        assert state.status == AgentStatus.COMPLETED

        # Verify the response mentions the timeframe
        final_message = None
        for step in reversed(state.steps):
            if step.step_type == StepType.ASSISTANT_MESSAGE:
                final_message = step.content
                break

        assert final_message is not None
        assert "5-7" in final_message or "business days" in final_message

    @pytest.mark.asyncio
    async def test_simple_workflow_scenario(self):
        """Test a simple multi-step workflow."""
        workflow = (
            WorkflowBuilder("research-workflow", "Research Workflow")
            .add_start("input")
            .add_transform("research", "passthrough")
            .add_transform("review", "passthrough")
            .add_end("complete")
            .connect("input", "research")
            .connect("research", "review")
            .connect("review", "complete")
            .build()
        )

        engine = WorkflowEngine()
        state = engine.launch(workflow)
        state = await engine.start(workflow, state, input_data={"topic": "AI agents"})
        state = await engine.run_to_completion(workflow, state)

        assert state.status == WorkflowStatus.COMPLETED
