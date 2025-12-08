"""
Unit tests for new workflow node handlers.
Tests HTTP Request, LLM, Code, and Knowledge Retrieval nodes.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.workflows.types import NodeType, NodeDefinition
from app.workflows.state import WorkflowState
from app.workflows.handlers import (
    HTTPRequestNodeHandler,
    LLMNodeHandler,
    CodeNodeHandler,
    KnowledgeRetrievalNodeHandler,
)
from app.workflows.reducer import WorkflowReducer


class TestHTTPRequestNodeHandler:
    """Tests for HTTP Request node handler."""

    @pytest.fixture
    def handler(self):
        return HTTPRequestNodeHandler()

    @pytest.fixture
    def state(self):
        state = WorkflowState(workflow_id="test", execution_id="exec-1")
        state.set_variable("user_id", "123")
        state.set_variable("api_key", "secret")
        return state

    @pytest.mark.asyncio
    async def test_get_request(self, handler, state):
        """Test making a GET request."""
        node = NodeDefinition(
            id="http-1",
            type=NodeType.HTTP_REQUEST,
            config={
                "url": "https://api.example.com/users",
                "method": "GET",
                "output_variable": "api_response",
            },
        )

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.headers = {"content-type": "application/json"}
            mock_response.json.return_value = {"users": [{"id": 1}]}
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client_class.return_value.__aenter__.return_value = mock_client

            result = await handler.execute(node, state, {})

            assert "api_response" in result
            assert result["api_response"]["status_code"] == 200
            assert result["api_response"]["body"] == {"users": [{"id": 1}]}

    @pytest.mark.asyncio
    async def test_post_request_with_json(self, handler, state):
        """Test making a POST request with JSON body."""
        node = NodeDefinition(
            id="http-1",
            type=NodeType.HTTP_REQUEST,
            config={
                "url": "https://api.example.com/users",
                "method": "POST",
                "body": {"name": "Test User"},
                "headers": {"Authorization": "Bearer token"},
            },
        )

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 201
            mock_response.headers = {"content-type": "application/json"}
            mock_response.json.return_value = {"id": 1, "name": "Test User"}
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_class.return_value.__aenter__.return_value = mock_client

            result = await handler.execute(node, state, {})

            assert result["response"]["status_code"] == 201
            mock_client.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_variable_substitution_in_url(self, handler, state):
        """Test variable substitution in URL."""
        node = NodeDefinition(
            id="http-1",
            type=NodeType.HTTP_REQUEST,
            config={
                "url": "https://api.example.com/users/{{user_id}}",
                "method": "GET",
            },
        )

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.headers = {}
            mock_response.json.return_value = {}
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client_class.return_value.__aenter__.return_value = mock_client

            await handler.execute(node, state, {})

            # Verify the URL was correctly substituted
            call_args = mock_client.get.call_args
            assert "users/123" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_text_response_fallback(self, handler, state):
        """Test fallback to text when JSON parsing fails."""
        node = NodeDefinition(
            id="http-1",
            type=NodeType.HTTP_REQUEST,
            config={
                "url": "https://example.com",
                "method": "GET",
            },
        )

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.headers = {"content-type": "text/html"}
            mock_response.json.side_effect = ValueError("Not JSON")
            mock_response.text = "<html>Hello</html>"
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client_class.return_value.__aenter__.return_value = mock_client

            result = await handler.execute(node, state, {})

            assert result["response"]["body"] == "<html>Hello</html>"


class TestLLMNodeHandler:
    """Tests for LLM node handler."""

    @pytest.fixture
    def handler(self):
        return LLMNodeHandler()

    @pytest.fixture
    def state(self):
        state = WorkflowState(workflow_id="test", execution_id="exec-1")
        state.set_variable("topic", "Python programming")
        return state

    @pytest.fixture
    def mock_llm_client(self):
        client = AsyncMock()
        client.chat = AsyncMock(return_value={
            "content": "Here is information about Python programming...",
            "usage": {"prompt_tokens": 10, "completion_tokens": 50},
        })
        return client

    @pytest.mark.asyncio
    async def test_llm_call(self, handler, state, mock_llm_client):
        """Test making an LLM call."""
        node = NodeDefinition(
            id="llm-1",
            type=NodeType.LLM,
            config={
                "prompt": "Tell me about {{topic}}",
                "model": "gpt-4",
                "temperature": 0.5,
                "max_tokens": 500,
                "output_variable": "llm_output",
            },
        )

        result = await handler.execute(node, state, {"llm_client": mock_llm_client})

        assert "llm_output" in result
        assert "Python programming" in result["llm_output"]
        mock_llm_client.chat.assert_called_once()

    @pytest.mark.asyncio
    async def test_llm_with_system_prompt(self, handler, state, mock_llm_client):
        """Test LLM call with system prompt."""
        node = NodeDefinition(
            id="llm-1",
            type=NodeType.LLM,
            config={
                "prompt": "Tell me about {{topic}}",
                "system_prompt": "You are a helpful coding assistant.",
                "model": "gpt-4",
            },
        )

        await handler.execute(node, state, {"llm_client": mock_llm_client})

        call_args = mock_llm_client.chat.call_args
        messages = call_args.kwargs["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    @pytest.mark.asyncio
    async def test_llm_without_client_raises(self, handler, state):
        """Test that missing LLM client raises error."""
        node = NodeDefinition(
            id="llm-1",
            type=NodeType.LLM,
            config={"prompt": "Hello"},
        )

        with pytest.raises(ValueError, match="LLM client not provided"):
            await handler.execute(node, state, {})


class TestCodeNodeHandler:
    """Tests for Code execution node handler."""

    @pytest.fixture
    def handler(self):
        return CodeNodeHandler()

    @pytest.fixture
    def state(self):
        state = WorkflowState(workflow_id="test", execution_id="exec-1")
        state.set_variable("numbers", [1, 2, 3, 4, 5])
        state.set_variable("factor", 2)
        return state

    @pytest.mark.asyncio
    async def test_basic_code_execution(self, handler, state):
        """Test basic Python code execution."""
        node = NodeDefinition(
            id="code-1",
            type=NodeType.CODE,
            config={
                "code": "result = sum(numbers) * factor",
                "output_variable": "calculation",
            },
        )

        result = await handler.execute(node, state, {})

        assert result["calculation"] == 30  # (1+2+3+4+5) * 2

    @pytest.mark.asyncio
    async def test_code_with_list_operations(self, handler, state):
        """Test code with list operations."""
        node = NodeDefinition(
            id="code-1",
            type=NodeType.CODE,
            config={
                # Use list() and map() instead of list comprehension
                # List comprehensions have scoping issues in exec()
                "code": "result = list(map(lambda x: x * factor, numbers))",
            },
        )

        result = await handler.execute(node, state, {})

        assert result["result"] == [2, 4, 6, 8, 10]

    @pytest.mark.asyncio
    async def test_code_error_handling(self, handler, state):
        """Test that code errors are caught."""
        node = NodeDefinition(
            id="code-1",
            type=NodeType.CODE,
            config={
                "code": "result = undefined_variable",
            },
        )

        result = await handler.execute(node, state, {})

        assert result["result"] is None
        assert "error" in result
        assert "undefined_variable" in result["error"]

    @pytest.mark.asyncio
    async def test_code_restricted_builtins(self, handler, state):
        """Test that dangerous builtins are restricted."""
        node = NodeDefinition(
            id="code-1",
            type=NodeType.CODE,
            config={
                # Try to use restricted function
                "code": "result = open('/etc/passwd').read()",
            },
        )

        result = await handler.execute(node, state, {})

        # Should fail because 'open' is not in allowed builtins
        assert "error" in result

    @pytest.mark.asyncio
    async def test_code_accesses_variables(self, handler, state):
        """Test that code can access workflow variables."""
        state.set_variable("name", "Alice")
        state.set_variable("age", 30)

        node = NodeDefinition(
            id="code-1",
            type=NodeType.CODE,
            config={
                "code": "result = f'{name} is {age} years old'",
            },
        )

        result = await handler.execute(node, state, {})

        assert result["result"] == "Alice is 30 years old"


class TestKnowledgeRetrievalNodeHandler:
    """Tests for Knowledge Retrieval node handler."""

    @pytest.fixture
    def handler(self):
        return KnowledgeRetrievalNodeHandler()

    @pytest.fixture
    def state(self):
        state = WorkflowState(workflow_id="test", execution_id="exec-1")
        state.set_variable("query", "How to use Python decorators?")
        return state

    @pytest.fixture
    def mock_retriever(self):
        retriever = AsyncMock()
        # Return mock search results
        mock_results = [
            MagicMock(
                document=MagicMock(content="Decorators are a way to modify functions...", metadata={"source": "doc1"}),
                score=0.95
            ),
            MagicMock(
                document=MagicMock(content="To create a decorator, use @...", metadata={"source": "doc2"}),
                score=0.85
            ),
        ]
        retriever.retrieve = AsyncMock(return_value=mock_results)
        return retriever

    @pytest.mark.asyncio
    async def test_knowledge_retrieval(self, handler, state, mock_retriever):
        """Test knowledge retrieval from vector store."""
        node = NodeDefinition(
            id="rag-1",
            type=NodeType.KNOWLEDGE_RETRIEVAL,
            config={
                "query_variable": "query",
                "top_k": 5,
                "output_variable": "docs",
            },
        )

        result = await handler.execute(node, state, {"retriever": mock_retriever})

        assert "docs" in result
        assert len(result["docs"]) == 2
        assert "Decorators" in result["docs"][0]["content"]
        assert result["docs"][0]["score"] == 0.95
        mock_retriever.retrieve.assert_called_once_with("How to use Python decorators?", top_k=5)

    @pytest.mark.asyncio
    async def test_retrieval_metadata(self, handler, state, mock_retriever):
        """Test that retrieval metadata is included."""
        node = NodeDefinition(
            id="rag-1",
            type=NodeType.KNOWLEDGE_RETRIEVAL,
            config={
                "query_variable": "query",
            },
        )

        result = await handler.execute(node, state, {"retriever": mock_retriever})

        assert "retrieval_metadata" in result
        assert result["retrieval_metadata"]["query"] == "How to use Python decorators?"
        assert result["retrieval_metadata"]["result_count"] == 2

    @pytest.mark.asyncio
    async def test_retrieval_without_retriever_raises(self, handler, state):
        """Test that missing retriever raises error."""
        node = NodeDefinition(
            id="rag-1",
            type=NodeType.KNOWLEDGE_RETRIEVAL,
            config={},
        )

        with pytest.raises(ValueError, match="Retriever not provided"):
            await handler.execute(node, state, {})


class TestWorkflowReducerWithNewNodes:
    """Test that new node types are registered in reducer."""

    def test_new_node_types_registered(self):
        """Test that all new node types have handlers."""
        reducer = WorkflowReducer()

        assert NodeType.HTTP_REQUEST in reducer._handlers
        assert NodeType.LLM in reducer._handlers
        assert NodeType.CODE in reducer._handlers
        assert NodeType.KNOWLEDGE_RETRIEVAL in reducer._handlers

    @pytest.mark.asyncio
    async def test_execute_http_node(self):
        """Test executing HTTP node through reducer."""
        from app.workflows.state import WorkflowDefinition

        reducer = WorkflowReducer()
        state = WorkflowState(workflow_id="test", execution_id="exec-1")
        definition = WorkflowDefinition(
            id="wf-1",
            name="Test",
            nodes=[
                NodeDefinition(
                    id="http-1",
                    type=NodeType.HTTP_REQUEST,
                    config={
                        "url": "https://api.example.com",
                        "method": "GET",
                    },
                ),
            ],
            edges=[],
        )

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.headers = {}
            mock_response.json.return_value = {"data": "test"}
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client_class.return_value.__aenter__.return_value = mock_client

            output = await reducer.execute_node(definition, state, "http-1", {})

            assert output["response"]["status_code"] == 200
