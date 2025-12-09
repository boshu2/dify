"""
Tracer bullet tests for core infrastructure.

These tests validate that each component works in isolation with minimal setup.
Each test is a simple "smoke test" that proves the feature path works.
"""
import io
import pytest
from unittest.mock import AsyncMock, MagicMock


class TestAPITracerBullet:
    """Validate FastAPI application loads and responds."""

    def test_app_imports(self):
        """App module imports without errors."""
        from app.main import app
        assert app is not None

    def test_app_has_routes(self):
        """App has expected routes registered."""
        from app.main import app
        routes = [r.path for r in app.routes]
        assert any("/health" in r for r in routes)

    @pytest.mark.asyncio
    async def test_health_endpoint(self):
        """Health endpoint responds."""
        from httpx import AsyncClient, ASGITransport
        from app.main import app

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/health")
            assert response.status_code == 200


class TestRedisTracerBullet:
    """Validate Redis/cache integration works."""

    def test_cache_imports(self):
        """Cache classes can be imported."""
        from app.core.cache import InMemoryCache, LLMResponseCache
        assert InMemoryCache is not None
        assert LLMResponseCache is not None

    @pytest.mark.asyncio
    async def test_in_memory_cache_operations(self):
        """In-memory cache set/get works."""
        from app.core.cache import InMemoryCache

        cache = InMemoryCache()
        await cache.set("test_key", "test_value", ttl_seconds=60)
        value = await cache.get("test_key")
        assert value == "test_value"


class TestPostgresTracerBullet:
    """Validate database models and operations."""

    def test_base_imports(self):
        """Database Base class imports."""
        from app.core.database import Base
        assert Base is not None

    def test_models_import(self):
        """Database models import without errors."""
        from app.models import Agent, DataSource, LLMProvider
        assert Agent is not None
        assert DataSource is not None
        assert LLMProvider is not None

    def test_session_factory_imports(self):
        """Session factory can be imported."""
        from app.core.database import get_db, async_session_maker
        assert get_db is not None
        assert async_session_maker is not None

    @pytest.mark.asyncio
    async def test_in_memory_sqlite(self):
        """Can create tables with SQLite for testing."""
        from sqlalchemy.ext.asyncio import create_async_engine
        from sqlalchemy import text
        from app.core.database import Base

        engine = create_async_engine("sqlite+aiosqlite:///:memory:")
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        # Verify tables created
        async with engine.connect() as conn:
            result = await conn.execute(
                text("SELECT name FROM sqlite_master WHERE type='table'")
            )
            tables = [r[0] for r in result.fetchall()]
            assert len(tables) > 0


class TestDocumentExtractorTracerBullet:
    """Validate document extractors work."""

    def test_extractors_import(self):
        """All extractors import without errors."""
        from app.extractors import (
            PDFExtractor,
            DocxExtractor,
            HTMLExtractor,
            SpreadsheetExtractor,
            get_extractor,
        )
        assert PDFExtractor is not None
        assert DocxExtractor is not None
        assert HTMLExtractor is not None
        assert SpreadsheetExtractor is not None
        assert get_extractor is not None

    def test_extractor_registry(self):
        """Registry returns correct extractors."""
        from app.extractors import get_extractor

        assert get_extractor("test.pdf") is not None
        assert get_extractor("test.docx") is not None
        assert get_extractor("test.html") is not None
        assert get_extractor("test.csv") is not None
        assert get_extractor("test.xlsx") is not None

    @pytest.mark.asyncio
    async def test_html_extraction(self):
        """HTML extractor extracts text."""
        from app.extractors import HTMLExtractor

        html = b"<html><body><h1>Title</h1><p>Content</p></body></html>"
        extractor = HTMLExtractor()
        result = await extractor.extract(io.BytesIO(html), "test.html")

        assert "Title" in result.text
        assert "Content" in result.text

    @pytest.mark.asyncio
    async def test_csv_extraction(self):
        """CSV extractor extracts data."""
        from app.extractors import SpreadsheetExtractor

        csv = b"Name,Age\nAlice,30\nBob,25"
        extractor = SpreadsheetExtractor()
        result = await extractor.extract(io.BytesIO(csv), "test.csv")

        assert "Alice" in result.text
        assert "30" in result.text


class TestWorkflowTracerBullet:
    """Validate workflow engine works."""

    def test_workflow_imports(self):
        """Workflow components import."""
        from app.workflows.types import NodeType, NodeDefinition
        from app.workflows.state import WorkflowState, WorkflowDefinition
        from app.workflows.reducer import WorkflowReducer

        assert NodeType is not None
        assert WorkflowReducer is not None

    def test_node_types_exist(self):
        """All expected node types exist."""
        from app.workflows.types import NodeType

        expected = ["START", "END", "AGENT", "CONDITION", "HTTP_REQUEST", "LLM", "CODE"]
        for name in expected:
            assert hasattr(NodeType, name), f"Missing NodeType.{name}"

    @pytest.mark.asyncio
    async def test_simple_workflow_execution(self):
        """Simple start->end workflow executes."""
        from app.workflows.types import NodeType, NodeDefinition
        from app.workflows.state import WorkflowState, WorkflowDefinition
        from app.workflows.reducer import WorkflowReducer

        definition = WorkflowDefinition(
            id="test-wf",
            name="Test",
            nodes=[
                NodeDefinition(id="start", type=NodeType.START, config={}),
                NodeDefinition(id="end", type=NodeType.END, config={}),
            ],
            edges=[{"source": "start", "target": "end"}],
        )

        state = WorkflowState(workflow_id="test-wf", execution_id="exec-1")
        reducer = WorkflowReducer()

        output = await reducer.execute_node(definition, state, "start", {})
        assert output is not None

    @pytest.mark.asyncio
    async def test_code_node_execution(self):
        """Code node executes Python."""
        from app.workflows.types import NodeType, NodeDefinition
        from app.workflows.state import WorkflowState
        from app.workflows.handlers import CodeNodeHandler

        node = NodeDefinition(
            id="code-1",
            type=NodeType.CODE,
            config={"code": "result = 2 + 2"},
        )
        state = WorkflowState(workflow_id="test", execution_id="exec-1")
        handler = CodeNodeHandler()

        output = await handler.execute(node, state, {})
        assert output["result"] == 4


class TestRAGTracerBullet:
    """Validate RAG pipeline components work."""

    def test_rag_imports(self):
        """RAG components import."""
        from app.rag import (
            VectorStore,
            InMemoryVectorStore,
            QdrantVectorStore,
            TextNormalizer,
        )
        from app.rag.chunker import FixedSizeChunker

        assert VectorStore is not None
        assert InMemoryVectorStore is not None
        assert QdrantVectorStore is not None
        assert TextNormalizer is not None
        assert FixedSizeChunker is not None

    @pytest.mark.asyncio
    async def test_in_memory_vector_store(self):
        """InMemoryVectorStore CRUD works."""
        from app.rag.vector_store import InMemoryVectorStore, Document

        store = InMemoryVectorStore()

        # Add
        doc = Document(id="doc1", content="Test content", embedding=[0.1, 0.2, 0.3])
        ids = await store.add_documents([doc])
        assert "doc1" in ids

        # Get
        retrieved = await store.get("doc1")
        assert retrieved is not None
        assert retrieved.content == "Test content"

        # Search
        results = await store.search([0.1, 0.2, 0.3], limit=1)
        assert len(results) == 1
        assert results[0].document.id == "doc1"

        # Delete
        deleted = await store.delete(["doc1"])
        assert deleted == 1
        assert await store.get("doc1") is None

    def test_fixed_size_chunker(self):
        """FixedSizeChunker splits text."""
        from app.rag.chunker import FixedSizeChunker

        chunker = FixedSizeChunker(chunk_size=50, chunk_overlap=10)
        text = "This is a test. " * 20  # ~320 chars
        chunks = chunker.chunk(text)

        assert len(chunks) > 1
        assert all(len(c.content) <= 60 for c in chunks)

    def test_text_normalizer(self):
        """TextNormalizer cleans text."""
        from app.rag import TextNormalizer

        normalizer = TextNormalizer(
            remove_extra_whitespace=True,
            remove_html_tags=True,
        )

        dirty = "  <p>Hello</p>   World  "
        clean = normalizer.normalize(dirty)

        assert "<p>" not in clean
        assert "Hello" in clean
        assert "World" in clean


class TestLLMProviderTracerBullet:
    """Validate LLM provider interfaces."""

    def test_llm_providers_import(self):
        """LLM providers import."""
        from app.providers.llm import (
            BaseLLMProvider,
            OpenAIProvider,
            AnthropicProvider,
            LLMProviderFactory,
        )

        assert BaseLLMProvider is not None
        assert OpenAIProvider is not None
        assert AnthropicProvider is not None
        assert LLMProviderFactory is not None

    @pytest.mark.asyncio
    async def test_mock_llm_chat(self):
        """Mock LLM responds to chat."""
        from app.agents.twelve_factor_agent import LLMClient

        # Create a simple mock
        class MockLLMClient(LLMClient):
            async def chat(self, messages, **kwargs):
                return {"content": "Mock response"}

            async def chat_stream(self, messages, **kwargs):
                yield {"content": "Mock stream"}

        client = MockLLMClient()
        response = await client.chat(
            messages=[{"role": "user", "content": "Hello"}],
            model="mock",
        )

        assert "content" in response
        assert len(response["content"]) > 0


class TestAgentTracerBullet:
    """Validate 12-factor agent works."""

    def test_agent_imports(self):
        """Agent components import."""
        from app.agents.twelve_factor_agent import Agent, AgentConfig, AgentState

        assert Agent is not None
        assert AgentConfig is not None
        assert AgentState is not None

    @pytest.mark.asyncio
    async def test_agent_launch(self):
        """Agent can launch with mock LLM."""
        from app.agents.twelve_factor_agent import Agent, AgentConfig, LLMClient

        class MockLLMClient(LLMClient):
            async def chat(self, messages, **kwargs):
                return {"content": "Hello!"}

            async def chat_stream(self, messages, **kwargs):
                yield {"content": "Hello!"}

        config = AgentConfig(
            agent_id="test-agent",
            purpose="Test agent",
            llm_client=MockLLMClient(),
        )
        agent = Agent(config)

        state = agent.launch("Hello, agent!")
        assert state is not None
        assert state.agent_id == "test-agent"


class TestTextUtilsTracerBullet:
    """Validate text utilities."""

    def test_utils_import(self):
        """Text utils import."""
        from app.rag.text_utils import (
            extract_sentences,
            extract_paragraphs,
            truncate_text,
            clean_markdown,
        )
        assert extract_sentences is not None

    def test_extract_sentences(self):
        """Sentence extraction works."""
        from app.rag.text_utils import extract_sentences

        text = "First sentence. Second sentence! Third sentence?"
        sentences = extract_sentences(text)
        assert len(sentences) >= 1

    def test_truncate_text(self):
        """Text truncation works."""
        from app.rag.text_utils import truncate_text

        long_text = "This is a very long text that should be truncated at some point."
        short = truncate_text(long_text, 20)

        assert len(short) <= 20
        assert short.endswith("...")

    def test_clean_markdown(self):
        """Markdown cleaning works."""
        from app.rag.text_utils import clean_markdown

        md = "# Title\n\n**Bold** and *italic* text.\n\n[Link](http://example.com)"
        clean = clean_markdown(md)

        assert "#" not in clean
        assert "**" not in clean
        assert "http://" not in clean
        assert "Title" in clean
        assert "Bold" in clean
