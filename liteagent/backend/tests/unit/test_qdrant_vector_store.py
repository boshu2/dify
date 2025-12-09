"""
Unit tests for Qdrant vector store integration.
Tests use in-memory mode for portability.
"""
import pytest
import uuid

from app.rag.vector_store import Document, QdrantVectorStore, SearchResult


# Fixed UUIDs for predictable testing
DOC1_ID = str(uuid.UUID("11111111-1111-1111-1111-111111111111"))
DOC2_ID = str(uuid.UUID("22222222-2222-2222-2222-222222222222"))
DOC3_ID = str(uuid.UUID("33333333-3333-3333-3333-333333333333"))


class TestQdrantVectorStore:
    """Tests for Qdrant vector store using in-memory mode."""

    @pytest.fixture
    async def store(self):
        """Create and initialize an in-memory Qdrant store."""
        store = QdrantVectorStore(
            collection_name="test_documents",
            embedding_dimension=4,  # Small dimension for testing
            in_memory=True,
        )
        await store.initialize()
        yield store
        await store.close()

    @pytest.fixture
    def sample_documents(self):
        """Create sample documents with embeddings."""
        return [
            Document(
                id=DOC1_ID,
                content="Python is a programming language.",
                embedding=[1.0, 0.0, 0.0, 0.0],
                metadata={"category": "programming", "source": "wiki"},
            ),
            Document(
                id=DOC2_ID,
                content="JavaScript is used for web development.",
                embedding=[0.9, 0.1, 0.0, 0.0],
                metadata={"category": "programming", "source": "tutorial"},
            ),
            Document(
                id=DOC3_ID,
                content="Machine learning uses neural networks.",
                embedding=[0.0, 0.0, 1.0, 0.0],
                metadata={"category": "ai", "source": "paper"},
            ),
        ]

    @pytest.mark.asyncio
    async def test_initialize_creates_collection(self, store):
        """Test that initialization creates the collection."""
        client = store._get_client()
        collections = client.get_collections().collections
        assert any(c.name == "test_documents" for c in collections)

    @pytest.mark.asyncio
    async def test_add_documents(self, store, sample_documents):
        """Test adding documents to the store."""
        ids = await store.add_documents(sample_documents)

        assert len(ids) == 3
        assert DOC1_ID in ids
        assert DOC2_ID in ids
        assert DOC3_ID in ids

    @pytest.mark.asyncio
    async def test_add_empty_list(self, store):
        """Test adding empty document list."""
        ids = await store.add_documents([])
        assert ids == []

    @pytest.mark.asyncio
    async def test_search_similar_documents(self, store, sample_documents):
        """Test searching for similar documents."""
        await store.add_documents(sample_documents)

        # Search with query similar to programming docs
        results = await store.search(
            query_embedding=[0.95, 0.05, 0.0, 0.0],
            limit=2,
        )

        assert len(results) == 2
        assert all(isinstance(r, SearchResult) for r in results)
        # Should return doc1 and doc2 (programming related)
        result_ids = [r.document.id for r in results]
        assert DOC1_ID in result_ids or DOC2_ID in result_ids

    @pytest.mark.asyncio
    async def test_search_with_limit(self, store, sample_documents):
        """Test search respects limit parameter."""
        await store.add_documents(sample_documents)

        results = await store.search(
            query_embedding=[0.5, 0.5, 0.5, 0.0],
            limit=1,
        )

        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_search_returns_scores(self, store, sample_documents):
        """Test that search returns similarity scores."""
        await store.add_documents(sample_documents)

        results = await store.search(
            query_embedding=[1.0, 0.0, 0.0, 0.0],
            limit=3,
        )

        # Check scores are present and in descending order
        assert all(r.score is not None for r in results)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    async def test_search_with_filter(self, store, sample_documents):
        """Test searching with metadata filter."""
        await store.add_documents(sample_documents)

        # Filter to only programming category
        results = await store.search(
            query_embedding=[0.5, 0.5, 0.5, 0.0],
            limit=10,
            filter={"category": "programming"},
        )

        assert len(results) == 2
        assert all(r.document.metadata.get("category") == "programming" for r in results)

    @pytest.mark.asyncio
    async def test_search_empty_embedding(self, store, sample_documents):
        """Test search with empty embedding returns empty."""
        await store.add_documents(sample_documents)

        results = await store.search(query_embedding=[], limit=10)
        assert results == []

    @pytest.mark.asyncio
    async def test_get_document(self, store, sample_documents):
        """Test retrieving a document by ID."""
        await store.add_documents(sample_documents)

        doc = await store.get(DOC1_ID)

        assert doc is not None
        assert doc.id == DOC1_ID
        assert doc.content == "Python is a programming language."
        assert doc.metadata["category"] == "programming"

    @pytest.mark.asyncio
    async def test_get_nonexistent_document(self, store):
        """Test retrieving nonexistent document returns None."""
        nonexistent_id = str(uuid.uuid4())
        doc = await store.get(nonexistent_id)
        assert doc is None

    @pytest.mark.asyncio
    async def test_delete_documents(self, store, sample_documents):
        """Test deleting documents by ID."""
        await store.add_documents(sample_documents)

        deleted_count = await store.delete([DOC1_ID, DOC2_ID])

        assert deleted_count == 2
        assert await store.get(DOC1_ID) is None
        assert await store.get(DOC2_ID) is None
        assert await store.get(DOC3_ID) is not None

    @pytest.mark.asyncio
    async def test_delete_empty_list(self, store):
        """Test deleting empty list returns 0."""
        deleted = await store.delete([])
        assert deleted == 0

    @pytest.mark.asyncio
    async def test_upsert_behavior(self, store):
        """Test that adding document with same ID updates it."""
        test_id = str(uuid.uuid4())
        doc1 = Document(
            id=test_id,
            content="Original content",
            embedding=[1.0, 0.0, 0.0, 0.0],
            metadata={"version": 1},
        )
        await store.add_documents([doc1])

        # Update with same ID
        doc1_updated = Document(
            id=test_id,
            content="Updated content",
            embedding=[1.0, 0.0, 0.0, 0.0],
            metadata={"version": 2},
        )
        await store.add_documents([doc1_updated])

        # Should have updated content
        retrieved = await store.get(test_id)
        assert retrieved.content == "Updated content"
        assert retrieved.metadata["version"] == 2

    @pytest.mark.asyncio
    async def test_document_with_embedding_retrieved(self, store):
        """Test that document embedding can be retrieved."""
        test_id = str(uuid.uuid4())
        doc = Document(
            id=test_id,
            content="Test",
            embedding=[0.1, 0.2, 0.3, 0.4],
        )
        await store.add_documents([doc])

        retrieved = await store.get(test_id)

        assert retrieved is not None
        # Qdrant normalizes vectors with cosine distance
        # Just check that embedding has correct length
        assert retrieved.embedding is not None
        assert len(retrieved.embedding) == 4

    @pytest.mark.asyncio
    async def test_search_rank_ordering(self, store, sample_documents):
        """Test that search results have correct rank ordering."""
        await store.add_documents(sample_documents)

        results = await store.search(
            query_embedding=[1.0, 0.0, 0.0, 0.0],
            limit=3,
        )

        ranks = [r.rank for r in results]
        assert ranks == [1, 2, 3]


class TestQdrantVectorStoreConfiguration:
    """Tests for Qdrant store configuration options."""

    def test_in_memory_mode(self):
        """Test in-memory mode initialization."""
        store = QdrantVectorStore(in_memory=True)
        client = store._get_client()
        assert client is not None

    def test_default_collection_name(self):
        """Test default collection name."""
        store = QdrantVectorStore(in_memory=True)
        assert store.collection_name == "documents"

    def test_custom_collection_name(self):
        """Test custom collection name."""
        store = QdrantVectorStore(
            collection_name="custom_collection",
            in_memory=True,
        )
        assert store.collection_name == "custom_collection"

    def test_custom_embedding_dimension(self):
        """Test custom embedding dimension."""
        store = QdrantVectorStore(
            embedding_dimension=1536,
            in_memory=True,
        )
        assert store.embedding_dimension == 1536

    @pytest.mark.asyncio
    async def test_close_client(self):
        """Test closing the client."""
        store = QdrantVectorStore(in_memory=True)
        store._get_client()  # Initialize client
        assert store._client is not None

        await store.close()
        assert store._client is None


class TestQdrantVectorStoreExport:
    """Tests for module exports."""

    def test_export_from_rag_module(self):
        """Test QdrantVectorStore is exported from rag module."""
        from app.rag import QdrantVectorStore as ExportedQdrant
        assert ExportedQdrant is QdrantVectorStore

    def test_export_inmemory_from_rag_module(self):
        """Test InMemoryVectorStore is exported from rag module."""
        from app.rag import InMemoryVectorStore
        assert InMemoryVectorStore is not None
