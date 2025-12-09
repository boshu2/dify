"""
Tests for RAG Pipeline Components.

Tests:
- Embeddings (Nemotron, OpenAI, NoEmbedder)
- Vector Store (pgvector, in-memory)
- Retriever (Vector, BM25, Hybrid)
- Chunker (Fixed, Semantic, Recursive)
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import json

from app.rag.embeddings import (
    EmbeddingProvider,
    EmbeddingResult,
    NemotronEmbedder,
    OpenAIEmbedder,
    NoEmbedder,
    create_embedder,
)
from app.rag.vector_store import (
    Document,
    SearchResult,
    VectorStore,
    InMemoryVectorStore,
    PgVectorStore,
)
from app.rag.retriever import (
    Retriever,
    RetrievalResult,
    VectorRetriever,
    BM25Retriever,
    HybridRetriever,
    create_retriever,
)
from app.rag.chunker import (
    Chunk,
    TextChunker,
    FixedSizeChunker,
    SemanticChunker,
    RecursiveChunker,
    create_chunker,
)


# =============================================================================
# Embedding Tests
# =============================================================================

class TestEmbeddingResult:
    """Test EmbeddingResult dataclass."""

    def test_embedding_result_creation(self):
        """Test creating embedding result."""
        result = EmbeddingResult(
            embeddings=[[0.1, 0.2, 0.3]],
            model="test-model",
            usage={"tokens": 10},
        )

        assert result.embeddings == [[0.1, 0.2, 0.3]]
        assert result.model == "test-model"
        assert result.usage["tokens"] == 10


class TestNoEmbedder:
    """Test NoEmbedder for BM25-only retrieval."""

    def test_no_embedder_dimension(self):
        """NoEmbedder should have 0 dimension."""
        embedder = NoEmbedder()
        assert embedder.dimension == 0

    @pytest.mark.asyncio
    async def test_no_embedder_returns_empty(self):
        """NoEmbedder should return empty embeddings."""
        embedder = NoEmbedder()
        result = await embedder.embed(["test text"])

        assert result.embeddings == [[]]
        assert result.model == "none"

    @pytest.mark.asyncio
    async def test_no_embedder_query(self):
        """NoEmbedder embed_query should return empty list."""
        embedder = NoEmbedder()
        embedding = await embedder.embed_query("test query")

        assert embedding == []


class TestNemotronEmbedder:
    """Test Nemotron embedder."""

    def test_nemotron_dimension(self):
        """Nemotron 1B should have 4096 dimensions."""
        embedder = NemotronEmbedder(api_key="test")
        assert embedder.dimension == 4096

    def test_nemotron_8b_dimension(self):
        """Nemotron 8B should have 4096 dimensions."""
        embedder = NemotronEmbedder(
            api_key="test",
            model=NemotronEmbedder.NEMOTRON_8B,
        )
        assert embedder.dimension == 4096

    @pytest.mark.asyncio
    async def test_nemotron_embed_format(self):
        """Nemotron should format request correctly."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "data": [{"embedding": [0.1] * 4096}],
                "model": "nvidia/llama-nemotron-embed-1b-v2",
                "usage": {"total_tokens": 5},
            }
            mock_response.raise_for_status = MagicMock()

            mock_instance = AsyncMock()
            mock_instance.post.return_value = mock_response
            mock_instance.__aenter__.return_value = mock_instance
            mock_instance.__aexit__.return_value = None
            mock_client.return_value = mock_instance

            embedder = NemotronEmbedder(api_key="test-key")
            result = await embedder.embed(["test text"])

            assert len(result.embeddings) == 1
            assert len(result.embeddings[0]) == 4096

    @pytest.mark.asyncio
    async def test_nemotron_embed_query_adds_instruction(self):
        """embed_query should add retrieval instruction."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "data": [{"embedding": [0.1] * 4096}],
                "model": "nvidia/llama-nemotron-embed-1b-v2",
                "usage": {"total_tokens": 5},
            }
            mock_response.raise_for_status = MagicMock()

            mock_instance = AsyncMock()
            mock_instance.post.return_value = mock_response
            mock_instance.__aenter__.return_value = mock_instance
            mock_instance.__aexit__.return_value = None
            mock_client.return_value = mock_instance

            embedder = NemotronEmbedder(api_key="test-key")
            embedding = await embedder.embed_query("find documents")

            # Check that post was called with instructed query
            call_args = mock_instance.post.call_args
            request_body = call_args.kwargs.get("json", {})
            input_text = request_body.get("input", [""])[0]
            assert "Instruct:" in input_text


class TestOpenAIEmbedder:
    """Test OpenAI embedder."""

    def test_openai_dimension(self):
        """OpenAI text-embedding-3-small has 1536 dims by default."""
        embedder = OpenAIEmbedder(api_key="test")
        assert embedder.dimension == 1536

    def test_openai_large_dimension(self):
        """OpenAI text-embedding-3-large has 3072 dims."""
        embedder = OpenAIEmbedder(
            api_key="test",
            model="text-embedding-3-large",
        )
        assert embedder.dimension == 3072


class TestCreateEmbedder:
    """Test embedder factory function."""

    def test_create_no_embedder(self):
        """Factory should create NoEmbedder for 'none'."""
        embedder = create_embedder("none")
        assert isinstance(embedder, NoEmbedder)

    def test_create_nemotron_embedder(self):
        """Factory should create NemotronEmbedder."""
        embedder = create_embedder("nemotron", api_key="test")
        assert isinstance(embedder, NemotronEmbedder)

    def test_create_openai_embedder(self):
        """Factory should create OpenAIEmbedder."""
        embedder = create_embedder("openai", api_key="test")
        assert isinstance(embedder, OpenAIEmbedder)

    def test_create_unknown_raises(self):
        """Factory should raise for unknown provider."""
        with pytest.raises(ValueError, match="Unknown embedding provider"):
            create_embedder("unknown")


# =============================================================================
# Vector Store Tests
# =============================================================================

class TestDocument:
    """Test Document dataclass."""

    def test_document_creation(self):
        """Test creating document."""
        doc = Document(
            id="doc-1",
            content="Test content",
            embedding=[0.1, 0.2],
            metadata={"source": "test"},
        )

        assert doc.id == "doc-1"
        assert doc.content == "Test content"
        assert doc.embedding == [0.1, 0.2]
        assert doc.metadata["source"] == "test"


class TestInMemoryVectorStore:
    """Test in-memory vector store."""

    @pytest.mark.asyncio
    async def test_add_and_search(self):
        """Test adding docs and searching."""
        store = InMemoryVectorStore()

        doc = Document(
            id="doc-1",
            content="Hello world",
            embedding=[1.0, 0.0, 0.0],
        )
        await store.add_documents([doc])

        results = await store.search([1.0, 0.0, 0.0], limit=1)

        assert len(results) == 1
        assert results[0].document.id == "doc-1"
        assert results[0].score > 0.99  # Should be almost 1.0 (same vector)

    @pytest.mark.asyncio
    async def test_search_by_similarity(self):
        """Test that search returns by similarity."""
        store = InMemoryVectorStore()

        docs = [
            Document(id="1", content="A", embedding=[1.0, 0.0, 0.0]),
            Document(id="2", content="B", embedding=[0.0, 1.0, 0.0]),
            Document(id="3", content="C", embedding=[0.0, 0.0, 1.0]),
        ]
        await store.add_documents(docs)

        # Query most similar to doc 2
        results = await store.search([0.1, 0.9, 0.1], limit=2)

        assert len(results) == 2
        assert results[0].document.id == "2"  # Most similar

    @pytest.mark.asyncio
    async def test_delete(self):
        """Test deleting documents."""
        store = InMemoryVectorStore()

        doc = Document(id="doc-1", content="Test", embedding=[1.0, 0.0, 0.0])
        await store.add_documents([doc])

        await store.delete(["doc-1"])

        results = await store.search([1.0, 0.0, 0.0], limit=1)
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_filter_by_metadata(self):
        """Test filtering by metadata."""
        store = InMemoryVectorStore()

        docs = [
            Document(id="1", content="A", embedding=[1.0, 0.0, 0.0], metadata={"type": "a"}),
            Document(id="2", content="B", embedding=[1.0, 0.0, 0.0], metadata={"type": "b"}),
        ]
        await store.add_documents(docs)

        results = await store.search(
            [1.0, 0.0, 0.0],
            limit=10,
            filter={"type": "a"},
        )

        assert len(results) == 1
        assert results[0].document.id == "1"


class TestPgVectorStore:
    """Test pgvector store (mocked)."""

    def test_pgvector_config(self):
        """Test pgvector configuration."""
        store = PgVectorStore(
            connection_string="postgresql://test",
            embedding_dimension=4096,
        )

        assert store.connection_string == "postgresql://test"
        assert store.embedding_dimension == 4096
        assert store.table_name == "documents"

    def test_pgvector_custom_table(self):
        """Test pgvector with custom table name."""
        store = PgVectorStore(
            connection_string="postgresql://test",
            table_name="custom_docs",
            embedding_dimension=1536,
        )

        assert store.table_name == "custom_docs"
        assert store.embedding_dimension == 1536


# =============================================================================
# Retriever Tests
# =============================================================================

class TestBM25Retriever:
    """Test BM25 keyword retriever."""

    def test_bm25_initialization(self):
        """Test BM25 initializes with documents."""
        docs = [
            Document(id="1", content="The quick brown fox"),
            Document(id="2", content="The lazy dog sleeps"),
            Document(id="3", content="Quick foxes are fast"),
        ]

        retriever = BM25Retriever(documents=docs)

        assert len(retriever._documents) == 3

    def test_bm25_add_documents(self):
        """Test adding documents to BM25."""
        retriever = BM25Retriever()
        docs = [
            Document(id="1", content="The quick brown fox"),
            Document(id="2", content="The lazy dog sleeps"),
        ]
        retriever.add_documents(docs)

        assert len(retriever._documents) == 2

    @pytest.mark.asyncio
    async def test_bm25_retrieval(self):
        """Test BM25 retrieves relevant documents."""
        docs = [
            Document(id="1", content="The quick brown fox jumps"),
            Document(id="2", content="The lazy dog sleeps all day"),
            Document(id="3", content="Python programming language"),
        ]

        retriever = BM25Retriever(documents=docs)

        result = await retriever.retrieve("fox jumping", limit=2)

        assert len(result.documents) >= 1
        # Doc about fox should be most relevant
        assert result.documents[0].document.id == "1"

    @pytest.mark.asyncio
    async def test_bm25_handles_empty_query(self):
        """Test BM25 handles empty query."""
        docs = [Document(id="1", content="Test content")]

        retriever = BM25Retriever(documents=docs)

        result = await retriever.retrieve("", limit=5)

        # Should return empty results for empty query
        assert isinstance(result.documents, list)
        assert len(result.documents) == 0


class TestVectorRetriever:
    """Test vector-based retriever."""

    @pytest.mark.asyncio
    async def test_vector_retriever_uses_embedder(self):
        """Test vector retriever uses embedder and store."""
        mock_embedder = AsyncMock()
        mock_embedder.embed_query.return_value = [1.0, 0.0, 0.0]
        mock_embedder.model_name = "test-model"

        mock_store = AsyncMock()
        mock_store.search.return_value = [
            SearchResult(
                document=Document(id="1", content="Test"),
                score=0.95,
                rank=1,
            ),
        ]

        retriever = VectorRetriever(
            vector_store=mock_store,
            embedder=mock_embedder,
        )

        result = await retriever.retrieve("test query", limit=5)

        mock_embedder.embed_query.assert_called_once_with("test query")
        mock_store.search.assert_called_once()
        assert len(result.documents) == 1
        assert result.strategy == "vector"


class TestHybridRetriever:
    """Test hybrid retriever combining vector and BM25."""

    @pytest.mark.asyncio
    async def test_hybrid_combines_results(self):
        """Test hybrid retriever combines vector and BM25 results."""
        # Mock vector retriever
        mock_vector = AsyncMock()
        mock_vector.retrieve.return_value = RetrievalResult(
            documents=[
                SearchResult(document=Document(id="1", content="Vector result 1"), score=0.9, rank=1),
                SearchResult(document=Document(id="2", content="Vector result 2"), score=0.8, rank=2),
            ],
            query="test query",
            strategy="vector",
        )

        # Mock BM25 retriever
        mock_bm25 = AsyncMock()
        mock_bm25.retrieve.return_value = RetrievalResult(
            documents=[
                SearchResult(document=Document(id="2", content="BM25 result 2"), score=0.85, rank=1),
                SearchResult(document=Document(id="3", content="BM25 result 3"), score=0.75, rank=2),
            ],
            query="test query",
            strategy="bm25",
        )

        retriever = HybridRetriever(
            vector_retriever=mock_vector,
            bm25_retriever=mock_bm25,
        )

        result = await retriever.retrieve("test query", limit=3)

        # Should have combined unique results
        doc_ids = [sr.document.id for sr in result.documents]
        assert "1" in doc_ids
        assert "2" in doc_ids
        assert "3" in doc_ids

    @pytest.mark.asyncio
    async def test_hybrid_rrf_scoring(self):
        """Test hybrid uses RRF scoring."""
        # Doc 2 ranks #1 in both lists, so should be highest overall
        mock_vector = AsyncMock()
        mock_vector.retrieve.return_value = RetrievalResult(
            documents=[
                SearchResult(document=Document(id="2", content="B"), score=0.9, rank=1),
                SearchResult(document=Document(id="1", content="A"), score=0.8, rank=2),
            ],
            query="query",
            strategy="vector",
        )

        mock_bm25 = AsyncMock()
        mock_bm25.retrieve.return_value = RetrievalResult(
            documents=[
                SearchResult(document=Document(id="2", content="B"), score=0.95, rank=1),
                SearchResult(document=Document(id="1", content="A"), score=0.7, rank=2),
            ],
            query="query",
            strategy="bm25",
        )

        retriever = HybridRetriever(
            vector_retriever=mock_vector,
            bm25_retriever=mock_bm25,
            rrf_k=60,
        )

        result = await retriever.retrieve("query", limit=2)

        # Doc 2 ranks #1 in both, so should be highest with RRF
        assert result.documents[0].document.id == "2"


class TestCreateRetriever:
    """Test retriever factory function."""

    def test_create_bm25_retriever(self):
        """Factory should create BM25 retriever."""
        retriever = create_retriever(strategy="bm25")
        assert isinstance(retriever, BM25Retriever)

    def test_create_vector_retriever(self):
        """Factory should create vector retriever."""
        mock_store = AsyncMock()
        mock_embedder = AsyncMock()

        retriever = create_retriever(
            strategy="vector",
            vector_store=mock_store,
            embedder=mock_embedder,
        )
        assert isinstance(retriever, VectorRetriever)

    def test_create_hybrid_retriever(self):
        """Factory should create hybrid retriever."""
        mock_store = AsyncMock()
        mock_embedder = AsyncMock()

        retriever = create_retriever(
            strategy="hybrid",
            vector_store=mock_store,
            embedder=mock_embedder,
        )
        assert isinstance(retriever, HybridRetriever)

    def test_create_unknown_raises(self):
        """Factory should raise for unknown strategy."""
        with pytest.raises(ValueError, match="Unknown retrieval strategy"):
            create_retriever(strategy="unknown")


# =============================================================================
# Chunker Tests
# =============================================================================

class TestChunk:
    """Test Chunk dataclass."""

    def test_chunk_to_document(self):
        """Test converting chunk to document."""
        chunk = Chunk(
            content="Test content",
            index=0,
            start_char=0,
            end_char=12,
            metadata={"key": "value"},
        )

        doc = chunk.to_document(source_id="source-1")

        assert doc.content == "Test content"
        assert doc.metadata["chunk_index"] == 0
        assert doc.metadata["source_id"] == "source-1"


class TestFixedSizeChunker:
    """Test fixed-size chunker."""

    def test_basic_chunking(self):
        """Test basic fixed-size chunking."""
        chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=20)
        text = "a" * 250

        chunks = chunker.chunk(text)

        assert len(chunks) >= 2

    def test_chunk_overlap(self):
        """Test that chunks overlap correctly."""
        chunker = FixedSizeChunker(chunk_size=10, chunk_overlap=3)
        text = "abcdefghijklmnopqrstuvwxyz"

        chunks = chunker.chunk(text)

        # Each chunk should share characters with the next
        for i in range(len(chunks) - 1):
            end_of_current = chunks[i].content[-3:]
            start_of_next = chunks[i + 1].content[:3]
            # Due to overlap logic, there should be some shared content
            assert chunks[i].end_char > chunks[i + 1].start_char

    def test_overlap_validation(self):
        """Test that overlap < chunk_size."""
        with pytest.raises(ValueError):
            FixedSizeChunker(chunk_size=10, chunk_overlap=10)

        with pytest.raises(ValueError):
            FixedSizeChunker(chunk_size=10, chunk_overlap=15)

    def test_empty_text(self):
        """Test handling empty text."""
        chunker = FixedSizeChunker()
        chunks = chunker.chunk("")

        assert chunks == []

    def test_metadata_propagation(self):
        """Test that metadata propagates to chunks."""
        chunker = FixedSizeChunker(chunk_size=50, chunk_overlap=10)
        text = "a" * 100

        chunks = chunker.chunk(text, metadata={"source": "test"})

        for chunk in chunks:
            assert chunk.metadata["source"] == "test"


class TestSemanticChunker:
    """Test semantic chunker."""

    def test_sentence_boundary_respect(self):
        """Test that chunker respects sentence boundaries."""
        chunker = SemanticChunker(chunk_size=100, min_chunk_size=10)
        text = "This is sentence one. This is sentence two. This is sentence three."

        chunks = chunker.chunk(text)

        # Chunks should not cut sentences in half
        for chunk in chunks:
            # Should not end with partial sentence (unless it's the whole text)
            if chunk.content != text:
                assert not chunk.content.endswith(" i")  # Not cut in "is"

    def test_paragraph_handling(self):
        """Test handling of paragraphs."""
        chunker = SemanticChunker(chunk_size=100, min_chunk_size=10)
        text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."

        chunks = chunker.chunk(text)

        # Should create reasonable chunks
        assert len(chunks) >= 1

    def test_long_sentence_splitting(self):
        """Test that very long sentences get split."""
        chunker = SemanticChunker(chunk_size=50, chunk_overlap=10, min_chunk_size=10)
        text = "a" * 200  # One very long "sentence"

        chunks = chunker.chunk(text)

        # Should split the long sentence
        assert len(chunks) > 1

    def test_min_chunk_size(self):
        """Test minimum chunk size is respected."""
        chunker = SemanticChunker(chunk_size=100, min_chunk_size=50)
        text = "Short. Another short."

        chunks = chunker.chunk(text)

        # Very short text might produce no chunks if below min
        # Or it gets combined
        for chunk in chunks:
            assert len(chunk.content) >= 10  # Some reasonable minimum


class TestRecursiveChunker:
    """Test recursive chunker."""

    def test_splits_on_paragraphs_first(self):
        """Test that recursive chunker tries paragraph splits first."""
        chunker = RecursiveChunker(chunk_size=100)
        text = "Para one content.\n\nPara two content.\n\nPara three content."

        chunks = chunker.chunk(text)

        assert len(chunks) >= 1

    def test_falls_back_to_lines(self):
        """Test fallback to line splits."""
        chunker = RecursiveChunker(chunk_size=50)
        text = "Line one content here.\nLine two content here.\nLine three content here."

        chunks = chunker.chunk(text)

        assert len(chunks) >= 1

    def test_handles_no_separators(self):
        """Test handling text with no separators."""
        chunker = RecursiveChunker(chunk_size=20)
        text = "a" * 50  # No separators

        chunks = chunker.chunk(text)

        # Should still chunk by characters
        assert len(chunks) >= 1


class TestCreateChunker:
    """Test chunker factory function."""

    def test_create_fixed_chunker(self):
        """Factory should create fixed chunker."""
        chunker = create_chunker("fixed", chunk_size=500)
        assert isinstance(chunker, FixedSizeChunker)
        assert chunker.chunk_size == 500

    def test_create_semantic_chunker(self):
        """Factory should create semantic chunker."""
        chunker = create_chunker("semantic", chunk_size=1000)
        assert isinstance(chunker, SemanticChunker)

    def test_create_recursive_chunker(self):
        """Factory should create recursive chunker."""
        chunker = create_chunker("recursive")
        assert isinstance(chunker, RecursiveChunker)

    def test_create_unknown_raises(self):
        """Factory should raise for unknown strategy."""
        with pytest.raises(ValueError):
            create_chunker("unknown")


# =============================================================================
# Integration Tests
# =============================================================================

class TestRAGPipelineIntegration:
    """Integration tests for RAG pipeline."""

    @pytest.mark.asyncio
    async def test_chunk_embed_store_retrieve(self):
        """Test full pipeline: chunk -> embed -> store -> retrieve."""
        # 1. Chunk text
        chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=20)
        text = "Python is a programming language. " * 10

        chunks = chunker.chunk(text, metadata={"source": "test.txt"})
        assert len(chunks) >= 1

        # 2. Convert to documents (skip embedding for this test)
        documents = [chunk.to_document(source_id="test") for chunk in chunks]

        # 3. Add fake embeddings for in-memory store test
        for i, doc in enumerate(documents):
            doc.embedding = [float(i) / max(len(documents), 1), 0.5, 0.5]

        # 4. Store in vector store
        store = InMemoryVectorStore()
        await store.add_documents(documents)

        # 5. Search
        results = await store.search([0.5, 0.5, 0.5], limit=3)

        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_bm25_no_embedding_pipeline(self):
        """Test BM25 pipeline without embeddings."""
        # Use NoEmbedder
        embedder = NoEmbedder()
        assert embedder.dimension == 0

        # Create documents without embeddings
        docs = [
            Document(id="1", content="Machine learning with Python"),
            Document(id="2", content="Web development with JavaScript"),
            Document(id="3", content="Python data science tutorial"),
        ]

        # Use BM25 retriever (no embeddings needed)
        retriever = BM25Retriever(documents=docs)

        result = await retriever.retrieve("Python tutorial", limit=2)

        # Should return Python-related docs
        contents = [sr.document.content for sr in result.documents]
        assert any("Python" in c for c in contents)


# =============================================================================
# Reranker Tests
# =============================================================================

from app.rag.retriever import (
    Reranker,
    CrossEncoderReranker,
    WeightedScoreReranker,
    RetrievalPipeline,
)


class TestCrossEncoderReranker:
    """Test cross-encoder reranker."""

    @pytest.mark.asyncio
    async def test_rerank_reorders_by_similarity(self):
        """Test that reranker reorders results by similarity score."""
        # Create mock embedder
        mock_embedder = AsyncMock()

        # Mock embed_query to return different embeddings
        # Query embedding
        query_emb = [1.0, 0.0, 0.0]
        # Doc embeddings - doc2 is most similar to query
        doc1_emb = [0.5, 0.5, 0.5]  # Less similar
        doc2_emb = [0.9, 0.1, 0.0]  # Most similar
        doc3_emb = [0.0, 1.0, 0.0]  # Least similar

        mock_embedder.embed_query = AsyncMock(
            side_effect=[query_emb, doc1_emb, doc2_emb, doc3_emb]
        )

        reranker = CrossEncoderReranker(embedder=mock_embedder)

        # Initial results (wrong order)
        initial_results = [
            SearchResult(document=Document(id="1", content="Doc 1"), score=0.9, rank=1),
            SearchResult(document=Document(id="2", content="Doc 2"), score=0.7, rank=2),
            SearchResult(document=Document(id="3", content="Doc 3"), score=0.5, rank=3),
        ]

        reranked = await reranker.rerank("query", initial_results)

        # Doc 2 should now be first (most similar to query)
        assert reranked[0].document.id == "2"
        assert reranked[0].rank == 1

    @pytest.mark.asyncio
    async def test_rerank_respects_limit(self):
        """Test that reranker respects limit parameter."""
        mock_embedder = AsyncMock()
        mock_embedder.embed_query = AsyncMock(return_value=[0.5, 0.5, 0.5])

        reranker = CrossEncoderReranker(embedder=mock_embedder)

        initial_results = [
            SearchResult(document=Document(id=str(i), content=f"Doc {i}"), score=0.5, rank=i)
            for i in range(10)
        ]

        reranked = await reranker.rerank("query", initial_results, limit=3)

        assert len(reranked) == 3

    @pytest.mark.asyncio
    async def test_rerank_score_threshold(self):
        """Test that reranker filters by score threshold."""
        mock_embedder = AsyncMock()

        # Return different embeddings for each call
        embeddings = [
            [1.0, 0.0, 0.0],  # Query
            [0.9, 0.1, 0.0],  # High similarity
            [0.0, 1.0, 0.0],  # Low similarity
        ]
        mock_embedder.embed_query = AsyncMock(side_effect=embeddings)

        reranker = CrossEncoderReranker(
            embedder=mock_embedder,
            score_threshold=0.5,
        )

        initial_results = [
            SearchResult(document=Document(id="1", content="Similar"), score=0.9, rank=1),
            SearchResult(document=Document(id="2", content="Different"), score=0.3, rank=2),
        ]

        reranked = await reranker.rerank("query", initial_results)

        # Only high similarity doc should pass threshold
        assert len(reranked) == 1
        assert reranked[0].document.id == "1"

    @pytest.mark.asyncio
    async def test_rerank_empty_results(self):
        """Test that reranker handles empty results."""
        mock_embedder = AsyncMock()
        reranker = CrossEncoderReranker(embedder=mock_embedder)

        reranked = await reranker.rerank("query", [])

        assert reranked == []


class TestWeightedScoreReranker:
    """Test weighted score reranker."""

    @pytest.mark.asyncio
    async def test_weighted_rerank_combines_signals(self):
        """Test that weighted reranker combines multiple signals."""
        reranker = WeightedScoreReranker(
            original_weight=0.5,
            recency_weight=0.3,
            length_weight=0.2,
            preferred_length=100,
        )

        # Doc 2 has lower original score but is more recent
        initial_results = [
            SearchResult(
                document=Document(
                    id="1",
                    content="x" * 100,
                    metadata={"created_at": 100},
                ),
                score=0.9,
                rank=1,
            ),
            SearchResult(
                document=Document(
                    id="2",
                    content="x" * 100,
                    metadata={"created_at": 200},
                ),
                score=0.7,
                rank=2,
            ),
        ]

        reranked = await reranker.rerank("query", initial_results)

        # Both should be present, possibly in different order
        assert len(reranked) == 2

    @pytest.mark.asyncio
    async def test_weighted_rerank_respects_limit(self):
        """Test that weighted reranker respects limit."""
        reranker = WeightedScoreReranker()

        initial_results = [
            SearchResult(
                document=Document(id=str(i), content=f"Doc {i}"),
                score=0.5,
                rank=i,
            )
            for i in range(10)
        ]

        reranked = await reranker.rerank("query", initial_results, limit=5)

        assert len(reranked) == 5

    @pytest.mark.asyncio
    async def test_weighted_rerank_length_scoring(self):
        """Test that length scoring prefers optimal length."""
        reranker = WeightedScoreReranker(
            original_weight=0.0,
            recency_weight=0.0,
            length_weight=1.0,
            preferred_length=100,
        )

        # Doc with optimal length should score higher
        initial_results = [
            SearchResult(
                document=Document(id="1", content="x" * 100),  # Optimal
                score=0.5,
                rank=1,
            ),
            SearchResult(
                document=Document(id="2", content="x" * 500),  # Too long
                score=0.5,
                rank=2,
            ),
        ]

        reranked = await reranker.rerank("query", initial_results)

        # Optimal length doc should be first
        assert reranked[0].document.id == "1"


class TestRetrievalPipeline:
    """Test two-stage retrieval pipeline."""

    @pytest.mark.asyncio
    async def test_pipeline_without_reranker(self):
        """Test pipeline passes through results without reranker."""
        mock_retriever = AsyncMock()
        mock_retriever.retrieve.return_value = RetrievalResult(
            documents=[
                SearchResult(document=Document(id="1", content="Doc 1"), score=0.9, rank=1),
            ],
            query="test",
            strategy="bm25",
        )

        pipeline = RetrievalPipeline(retriever=mock_retriever)

        result = await pipeline.retrieve("test", limit=5)

        assert len(result.documents) == 1
        assert result.strategy == "bm25"

    @pytest.mark.asyncio
    async def test_pipeline_with_reranker(self):
        """Test pipeline applies reranker."""
        mock_retriever = AsyncMock()
        mock_retriever.retrieve.return_value = RetrievalResult(
            documents=[
                SearchResult(document=Document(id="1", content="Doc 1"), score=0.9, rank=1),
                SearchResult(document=Document(id="2", content="Doc 2"), score=0.7, rank=2),
            ],
            query="test",
            strategy="bm25",
        )

        mock_reranker = AsyncMock()
        mock_reranker.rerank.return_value = [
            SearchResult(document=Document(id="2", content="Doc 2"), score=0.95, rank=1),
            SearchResult(document=Document(id="1", content="Doc 1"), score=0.85, rank=2),
        ]

        pipeline = RetrievalPipeline(
            retriever=mock_retriever,
            reranker=mock_reranker,
            initial_limit_multiplier=2,
        )

        result = await pipeline.retrieve("test", limit=2)

        # Reranker was called
        mock_reranker.rerank.assert_called_once()

        # Strategy indicates reranking
        assert "rerank" in result.strategy

        # Metadata shows initial candidates
        assert result.metadata["reranked"] is True

    @pytest.mark.asyncio
    async def test_pipeline_initial_limit_multiplier(self):
        """Test that pipeline fetches more results for reranking."""
        mock_retriever = AsyncMock()
        mock_retriever.retrieve.return_value = RetrievalResult(
            documents=[],
            query="test",
            strategy="bm25",
        )

        mock_reranker = AsyncMock()
        mock_reranker.rerank.return_value = []

        pipeline = RetrievalPipeline(
            retriever=mock_retriever,
            reranker=mock_reranker,
            initial_limit_multiplier=3,
        )

        await pipeline.retrieve("test", limit=10)

        # Should request 3x the limit for initial retrieval
        mock_retriever.retrieve.assert_called_once()
        call_args = mock_retriever.retrieve.call_args
        assert call_args.kwargs.get("limit") == 30  # 10 * 3
