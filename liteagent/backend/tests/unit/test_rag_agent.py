"""
Tests for RAG-Enabled Agent Integration.

Tests the integration between 12-factor agent and RAG pipeline.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock

from app.agents.rag_agent import (
    RAGAgent,
    RAGAgentConfig,
    RAGPipelineBuilder,
    create_rag_agent,
)
from app.agents.twelve_factor_agent import (
    AgentStatus,
    LLMClient,
    ToolDefinition,
)
from app.rag.vector_store import Document, SearchResult
from app.rag.retriever import RetrievalResult


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


class TestRAGAgentConfig:
    """Test RAGAgentConfig."""

    def test_config_creation(self):
        """Test creating RAG agent config."""
        mock_retriever = AsyncMock()

        config = RAGAgentConfig(
            agent_id="test-agent",
            purpose="Answer questions",
            retriever=mock_retriever,
            top_k=10,
        )

        assert config.agent_id == "test-agent"
        assert config.purpose == "Answer questions"
        assert config.top_k == 10

    def test_config_defaults(self):
        """Test default config values."""
        mock_retriever = AsyncMock()

        config = RAGAgentConfig(
            agent_id="test",
            purpose="Test",
            retriever=mock_retriever,
        )

        assert config.top_k == 5
        assert config.include_sources is True
        assert config.max_iterations == 10


class TestRAGAgent:
    """Test RAGAgent class."""

    def test_agent_has_knowledge_search_tool(self):
        """Agent should have knowledge_search tool."""
        mock_retriever = AsyncMock()

        agent = RAGAgent(RAGAgentConfig(
            agent_id="test",
            purpose="Test",
            retriever=mock_retriever,
        ))

        # Check that knowledge_search tool exists
        tool_names = [t.name for t in agent._agent.config.tools]
        assert "knowledge_search" in tool_names

    def test_agent_purpose_includes_rag_instructions(self):
        """Agent purpose should include RAG instructions."""
        mock_retriever = AsyncMock()

        agent = RAGAgent(RAGAgentConfig(
            agent_id="test",
            purpose="Answer user questions",
            retriever=mock_retriever,
        ))

        purpose = agent._agent.config.purpose
        assert "Answer user questions" in purpose
        assert "knowledge base" in purpose
        assert "knowledge_search" in purpose

    def test_launch_creates_running_state(self):
        """Launch should create running state."""
        mock_retriever = AsyncMock()

        agent = RAGAgent(RAGAgentConfig(
            agent_id="test",
            purpose="Test",
            retriever=mock_retriever,
        ))

        state = agent.launch("Hello")

        assert state.status == AgentStatus.RUNNING
        assert len(state.steps) == 1

    @pytest.mark.asyncio
    async def test_knowledge_search_calls_retriever(self):
        """knowledge_search should call retriever."""
        mock_retriever = AsyncMock()
        mock_retriever.retrieve.return_value = RetrievalResult(
            documents=[
                SearchResult(
                    document=Document(id="1", content="Test content"),
                    score=0.9,
                    rank=1,
                ),
            ],
            query="test query",
            strategy="hybrid",
        )

        agent = RAGAgent(RAGAgentConfig(
            agent_id="test",
            purpose="Test",
            retriever=mock_retriever,
        ))

        result = await agent._search_knowledge("test query")

        mock_retriever.retrieve.assert_called_once()
        assert result["num_results"] == 1
        assert result["results"][0]["content"] == "Test content"

    @pytest.mark.asyncio
    async def test_knowledge_search_includes_sources(self):
        """knowledge_search should include source info when configured."""
        mock_retriever = AsyncMock()
        mock_retriever.retrieve.return_value = RetrievalResult(
            documents=[
                SearchResult(
                    document=Document(
                        id="doc-1",
                        content="Content",
                        metadata={"source": "file.txt"},
                    ),
                    score=0.8,
                    rank=1,
                ),
            ],
            query="query",
            strategy="vector",
        )

        agent = RAGAgent(RAGAgentConfig(
            agent_id="test",
            purpose="Test",
            retriever=mock_retriever,
            include_sources=True,
        ))

        result = await agent._search_knowledge("query")

        assert result["results"][0]["source"] == "file.txt"
        assert result["results"][0]["id"] == "doc-1"

    @pytest.mark.asyncio
    async def test_knowledge_search_handles_errors(self):
        """knowledge_search should handle retriever errors gracefully."""
        mock_retriever = AsyncMock()
        mock_retriever.retrieve.side_effect = Exception("Connection failed")

        agent = RAGAgent(RAGAgentConfig(
            agent_id="test",
            purpose="Test",
            retriever=mock_retriever,
        ))

        result = await agent._search_knowledge("query")

        assert "error" in result
        assert result["num_results"] == 0
        assert result["results"] == []

    @pytest.mark.asyncio
    async def test_agent_uses_knowledge_in_response(self):
        """Agent should use knowledge from RAG in its response."""
        mock_retriever = AsyncMock()
        mock_retriever.retrieve.return_value = RetrievalResult(
            documents=[
                SearchResult(
                    document=Document(id="1", content="Python 3.12 was released in 2023."),
                    score=0.95,
                    rank=1,
                ),
            ],
            query="python version",
            strategy="hybrid",
        )

        # LLM first calls knowledge_search, then responds
        mock_responses = [
            {
                "choices": [{
                    "message": {
                        "tool_calls": [{
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "knowledge_search",
                                "arguments": '{"query": "python latest version"}',
                            },
                        }],
                    },
                }],
            },
            {
                "choices": [{
                    "message": {
                        "content": "According to my knowledge base, Python 3.12 was released in 2023.",
                    },
                }],
            },
        ]

        agent = RAGAgent(RAGAgentConfig(
            agent_id="test",
            purpose="Answer Python questions",
            retriever=mock_retriever,
            llm_client=MockLLMClient(mock_responses),
        ))

        state = agent.launch("What is the latest Python version?")
        state = await agent.run_to_completion(state)

        assert state.status == AgentStatus.COMPLETED
        mock_retriever.retrieve.assert_called_once()

    def test_pause_and_resume(self):
        """Agent should support pause and resume."""
        mock_retriever = AsyncMock()

        agent = RAGAgent(RAGAgentConfig(
            agent_id="test",
            purpose="Test",
            retriever=mock_retriever,
        ))

        state = agent.launch("Test")
        state = agent.pause(state)
        assert state.status == AgentStatus.PAUSED

        state = agent.resume(state)
        assert state.status == AgentStatus.RUNNING

    def test_additional_tools_preserved(self):
        """Additional tools should be preserved alongside RAG tool."""
        mock_retriever = AsyncMock()

        custom_tool = ToolDefinition(
            name="custom_tool",
            description="Custom tool",
            parameters={},
        )

        agent = RAGAgent(RAGAgentConfig(
            agent_id="test",
            purpose="Test",
            retriever=mock_retriever,
            tools=[custom_tool],
        ))

        tool_names = [t.name for t in agent._agent.config.tools]
        assert "knowledge_search" in tool_names
        assert "custom_tool" in tool_names
        assert "request_human_input" in tool_names  # From 12-factor agent


class TestRAGPipelineBuilder:
    """Test RAGPipelineBuilder."""

    def test_builder_chain(self):
        """Builder should support method chaining."""
        mock_embedder = AsyncMock()
        mock_store = AsyncMock()

        builder = (
            RAGPipelineBuilder()
            .with_embedder(mock_embedder)
            .with_vector_store(mock_store)
        )

        assert builder._embedder is mock_embedder
        assert builder._vector_store is mock_store

    def test_add_documents(self):
        """Builder should accumulate documents."""
        docs1 = [Document(id="1", content="A")]
        docs2 = [Document(id="2", content="B")]

        builder = (
            RAGPipelineBuilder()
            .with_documents(docs1)
            .with_documents(docs2)
        )

        assert len(builder._documents) == 2

    @pytest.mark.asyncio
    async def test_build_bm25_retriever(self):
        """Builder should create BM25 retriever."""
        from app.rag.retriever import BM25Retriever

        docs = [
            Document(id="1", content="Test document one"),
            Document(id="2", content="Test document two"),
        ]

        builder = RAGPipelineBuilder().with_documents(docs)
        retriever = await builder.build_retriever(strategy="bm25")

        assert isinstance(retriever, BM25Retriever)


class TestCreateRagAgent:
    """Test create_rag_agent factory function."""

    @pytest.mark.asyncio
    async def test_create_basic_agent(self):
        """Factory should create basic RAG agent."""
        mock_retriever = AsyncMock()

        agent = await create_rag_agent(
            agent_id="test",
            purpose="Answer questions",
            retriever=mock_retriever,
        )

        assert isinstance(agent, RAGAgent)
        assert agent.config.agent_id == "test"

    @pytest.mark.asyncio
    async def test_create_agent_with_options(self):
        """Factory should accept additional options."""
        mock_retriever = AsyncMock()
        mock_llm = MockLLMClient()

        agent = await create_rag_agent(
            agent_id="test",
            purpose="Help users",
            retriever=mock_retriever,
            llm_client=mock_llm,
            top_k=10,
            max_iterations=5,
        )

        assert agent.config.top_k == 10
        assert agent.config.max_iterations == 5


class TestRAGAgentIntegration:
    """Integration tests for RAG agent."""

    @pytest.mark.asyncio
    async def test_full_rag_conversation(self):
        """Test complete RAG conversation flow."""
        # Set up retriever with test documents
        from app.rag.retriever import BM25Retriever
        from app.rag.vector_store import Document

        docs = [
            Document(id="doc1", content="The capital of France is Paris.", metadata={"source": "geography.txt"}),
            Document(id="doc2", content="Python is a programming language.", metadata={"source": "tech.txt"}),
            Document(id="doc3", content="Machine learning uses neural networks.", metadata={"source": "ai.txt"}),
        ]

        retriever = BM25Retriever(documents=docs)

        # LLM flow: search -> respond
        mock_responses = [
            {
                "choices": [{
                    "message": {
                        "tool_calls": [{
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "knowledge_search",
                                "arguments": '{"query": "capital of France"}',
                            },
                        }],
                    },
                }],
            },
            {
                "choices": [{
                    "message": {
                        "content": "Based on my search, the capital of France is Paris.",
                    },
                }],
            },
        ]

        agent = RAGAgent(RAGAgentConfig(
            agent_id="geography-assistant",
            purpose="Answer geography questions",
            retriever=retriever,
            llm_client=MockLLMClient(mock_responses),
        ))

        state = agent.launch("What is the capital of France?")
        state = await agent.run_to_completion(state)

        assert state.status == AgentStatus.COMPLETED

        # Verify the flow included tool usage
        from app.agents.twelve_factor_agent import StepType
        step_types = [s.step_type for s in state.steps]
        assert StepType.TOOL_CALL in step_types
        assert StepType.TOOL_RESULT in step_types
        assert StepType.ASSISTANT_MESSAGE in step_types

    @pytest.mark.asyncio
    async def test_rag_with_no_results(self):
        """Agent should handle no search results gracefully."""
        from app.rag.retriever import BM25Retriever

        # Empty document set
        retriever = BM25Retriever(documents=[])

        mock_responses = [
            {
                "choices": [{
                    "message": {
                        "tool_calls": [{
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "knowledge_search",
                                "arguments": '{"query": "unknown topic"}',
                            },
                        }],
                    },
                }],
            },
            {
                "choices": [{
                    "message": {
                        "content": "I couldn't find any information about that topic.",
                    },
                }],
            },
        ]

        agent = RAGAgent(RAGAgentConfig(
            agent_id="test",
            purpose="Test",
            retriever=retriever,
            llm_client=MockLLMClient(mock_responses),
        ))

        state = agent.launch("Tell me about quantum computing")
        state = await agent.run_to_completion(state)

        assert state.status == AgentStatus.COMPLETED
