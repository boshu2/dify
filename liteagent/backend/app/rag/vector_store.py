"""
Vector store implementations for RAG pipeline.

Supports:
- pgvector (PostgreSQL with vector extension)
- In-memory (for testing)
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
import json
import uuid


@dataclass
class Document:
    """A document chunk with embedding."""
    id: str
    content: str
    embedding: list[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "embedding": self.embedding,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class SearchResult:
    """Result from similarity search."""
    document: Document
    score: float  # Higher = more similar
    rank: int


class VectorStore(ABC):
    """Abstract vector store interface."""

    @abstractmethod
    async def add_documents(self, documents: list[Document]) -> list[str]:
        """
        Add documents to the store.

        Returns:
            List of document IDs.
        """
        pass

    @abstractmethod
    async def search(
        self,
        query_embedding: list[float],
        limit: int = 10,
        filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """
        Search for similar documents.

        Args:
            query_embedding: Query vector.
            limit: Max results to return.
            filter: Metadata filter.

        Returns:
            List of search results ordered by similarity.
        """
        pass

    @abstractmethod
    async def delete(self, document_ids: list[str]) -> int:
        """Delete documents by ID. Returns count deleted."""
        pass

    @abstractmethod
    async def get(self, document_id: str) -> Document | None:
        """Get a document by ID."""
        pass


class PgVectorStore(VectorStore):
    """
    PostgreSQL with pgvector extension.

    Requires:
    - PostgreSQL 15+
    - pgvector extension installed
    - asyncpg for async operations

    Table schema:
    CREATE EXTENSION IF NOT EXISTS vector;

    CREATE TABLE documents (
        id UUID PRIMARY KEY,
        content TEXT NOT NULL,
        embedding vector(4096),  -- Adjust dimension for your model
        metadata JSONB DEFAULT '{}',
        created_at TIMESTAMPTZ DEFAULT NOW()
    );

    CREATE INDEX ON documents USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100);
    """

    def __init__(
        self,
        connection_string: str,
        table_name: str = "documents",
        embedding_dimension: int = 4096,
    ):
        """
        Initialize pgvector store.

        Args:
            connection_string: PostgreSQL connection string.
            table_name: Table to store documents.
            embedding_dimension: Dimension of embeddings.
        """
        self.connection_string = connection_string
        self.table_name = table_name
        self.embedding_dimension = embedding_dimension
        self._pool = None

    async def _get_pool(self):
        """Get or create connection pool."""
        if self._pool is None:
            import asyncpg
            self._pool = await asyncpg.create_pool(
                self.connection_string,
                min_size=2,
                max_size=10,
            )
        return self._pool

    async def initialize(self) -> None:
        """
        Initialize the vector store.
        Creates table and indexes if they don't exist.
        """
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            # Enable pgvector extension
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

            # Create table
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    id UUID PRIMARY KEY,
                    content TEXT NOT NULL,
                    embedding vector({self.embedding_dimension}),
                    metadata JSONB DEFAULT '{{}}',
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)

            # Create index for similarity search
            # IVFFlat index for approximate nearest neighbor
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS {self.table_name}_embedding_idx
                ON {self.table_name}
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100)
            """)

    async def add_documents(self, documents: list[Document]) -> list[str]:
        """Add documents to pgvector."""
        if not documents:
            return []

        pool = await self._get_pool()
        ids = []

        async with pool.acquire() as conn:
            for doc in documents:
                doc_id = doc.id or str(uuid.uuid4())

                # Format embedding for pgvector
                embedding_str = None
                if doc.embedding:
                    embedding_str = f"[{','.join(map(str, doc.embedding))}]"

                await conn.execute(f"""
                    INSERT INTO {self.table_name} (id, content, embedding, metadata, created_at)
                    VALUES ($1, $2, $3::vector, $4, $5)
                    ON CONFLICT (id) DO UPDATE SET
                        content = EXCLUDED.content,
                        embedding = EXCLUDED.embedding,
                        metadata = EXCLUDED.metadata
                """,
                    uuid.UUID(doc_id),
                    doc.content,
                    embedding_str,
                    json.dumps(doc.metadata),
                    doc.created_at,
                )
                ids.append(doc_id)

        return ids

    async def search(
        self,
        query_embedding: list[float],
        limit: int = 10,
        filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """
        Search using cosine similarity.

        pgvector supports:
        - <-> : L2 distance
        - <#> : Inner product (negative)
        - <=> : Cosine distance

        We use cosine distance and convert to similarity.
        """
        if not query_embedding:
            return []

        pool = await self._get_pool()
        embedding_str = f"[{','.join(map(str, query_embedding))}]"

        # Build filter clause
        filter_clause = ""
        if filter:
            conditions = []
            for key, value in filter.items():
                conditions.append(f"metadata->'{key}' = '{json.dumps(value)}'")
            if conditions:
                filter_clause = "WHERE " + " AND ".join(conditions)

        async with pool.acquire() as conn:
            rows = await conn.fetch(f"""
                SELECT
                    id,
                    content,
                    embedding::text,
                    metadata,
                    created_at,
                    1 - (embedding <=> $1::vector) as similarity
                FROM {self.table_name}
                {filter_clause}
                ORDER BY embedding <=> $1::vector
                LIMIT $2
            """, embedding_str, limit)

        results = []
        for i, row in enumerate(rows):
            # Parse embedding back from text
            embedding = None
            if row["embedding"]:
                embedding_text = row["embedding"].strip("[]")
                if embedding_text:
                    embedding = [float(x) for x in embedding_text.split(",")]

            doc = Document(
                id=str(row["id"]),
                content=row["content"],
                embedding=embedding,
                metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                created_at=row["created_at"],
            )
            results.append(SearchResult(
                document=doc,
                score=float(row["similarity"]),
                rank=i + 1,
            ))

        return results

    async def delete(self, document_ids: list[str]) -> int:
        """Delete documents by ID."""
        if not document_ids:
            return 0

        pool = await self._get_pool()

        async with pool.acquire() as conn:
            result = await conn.execute(f"""
                DELETE FROM {self.table_name}
                WHERE id = ANY($1::uuid[])
            """, [uuid.UUID(id) for id in document_ids])

        # Parse "DELETE n" to get count
        return int(result.split()[-1])

    async def get(self, document_id: str) -> Document | None:
        """Get document by ID."""
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            row = await conn.fetchrow(f"""
                SELECT id, content, embedding::text, metadata, created_at
                FROM {self.table_name}
                WHERE id = $1
            """, uuid.UUID(document_id))

        if not row:
            return None

        embedding = None
        if row["embedding"]:
            embedding_text = row["embedding"].strip("[]")
            if embedding_text:
                embedding = [float(x) for x in embedding_text.split(",")]

        return Document(
            id=str(row["id"]),
            content=row["content"],
            embedding=embedding,
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
            created_at=row["created_at"],
        )

    async def close(self) -> None:
        """Close connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None


class InMemoryVectorStore(VectorStore):
    """
    In-memory vector store for testing.
    Uses brute-force cosine similarity.
    """

    def __init__(self):
        self._documents: dict[str, Document] = {}

    async def add_documents(self, documents: list[Document]) -> list[str]:
        ids = []
        for doc in documents:
            doc_id = doc.id or str(uuid.uuid4())
            doc.id = doc_id
            self._documents[doc_id] = doc
            ids.append(doc_id)
        return ids

    async def search(
        self,
        query_embedding: list[float],
        limit: int = 10,
        filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        if not query_embedding:
            return []

        results = []
        for doc in self._documents.values():
            # Apply filter
            if filter:
                skip = False
                for key, value in filter.items():
                    if doc.metadata.get(key) != value:
                        skip = True
                        break
                if skip:
                    continue

            if doc.embedding:
                score = self._cosine_similarity(query_embedding, doc.embedding)
                results.append((doc, score))

        # Sort by similarity descending
        results.sort(key=lambda x: x[1], reverse=True)

        return [
            SearchResult(document=doc, score=score, rank=i + 1)
            for i, (doc, score) in enumerate(results[:limit])
        ]

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if len(a) != len(b):
            return 0.0

        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    async def delete(self, document_ids: list[str]) -> int:
        count = 0
        for doc_id in document_ids:
            if doc_id in self._documents:
                del self._documents[doc_id]
                count += 1
        return count

    async def get(self, document_id: str) -> Document | None:
        return self._documents.get(document_id)
