"""RAG retrieval routes."""
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from typing import Any

from app.rag.vector_store import Document, InMemoryVectorStore
from app.rag.retriever import BM25Retriever, create_retriever
from app.rag.chunker import create_chunker

router = APIRouter()


# In-memory store for demo (in production, use persistent store)
_demo_store = InMemoryVectorStore()
_demo_documents: list[Document] = []


class DocumentCreate(BaseModel):
    """Document creation request."""
    content: str
    metadata: dict[str, Any] = {}


class DocumentResponse(BaseModel):
    """Document response."""
    id: str
    content: str
    metadata: dict[str, Any]


class SearchRequest(BaseModel):
    """Search request."""
    query: str
    limit: int = 5
    strategy: str = "bm25"  # "bm25", "vector", or "hybrid"
    min_score: float = 0.0


class SearchResult(BaseModel):
    """Search result."""
    document: DocumentResponse
    score: float
    rank: int


class SearchResponse(BaseModel):
    """Search response."""
    query: str
    strategy: str
    results: list[SearchResult]
    total_documents: int


class ChunkRequest(BaseModel):
    """Chunk text request."""
    text: str
    strategy: str = "fixed"  # "fixed", "semantic", or "recursive"
    chunk_size: int = 500
    chunk_overlap: int = 50


class ChunkResponse(BaseModel):
    """Chunk response."""
    chunks: list[dict[str, Any]]
    total_chunks: int
    strategy: str


@router.post("/documents", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def add_document(data: DocumentCreate):
    """Add a document to the retrieval index."""
    import uuid

    doc = Document(
        id=str(uuid.uuid4()),
        content=data.content,
        metadata=data.metadata,
    )

    _demo_documents.append(doc)

    return DocumentResponse(
        id=doc.id,
        content=doc.content,
        metadata=doc.metadata,
    )


@router.get("/documents", response_model=list[DocumentResponse])
async def list_documents(limit: int = 100, offset: int = 0):
    """List all documents in the index."""
    docs = _demo_documents[offset:offset + limit]
    return [
        DocumentResponse(id=d.id, content=d.content, metadata=d.metadata)
        for d in docs
    ]


@router.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete a document from the index."""
    global _demo_documents

    for i, doc in enumerate(_demo_documents):
        if doc.id == document_id:
            _demo_documents.pop(i)
            return {"status": "deleted"}

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="Document not found",
    )


@router.post("/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    """
    Search documents using various strategies.

    Strategies:
    - bm25: Keyword-based search (no embeddings required)
    - vector: Semantic similarity search (requires embeddings)
    - hybrid: Combination of BM25 and vector search
    """
    if not _demo_documents:
        return SearchResponse(
            query=request.query,
            strategy=request.strategy,
            results=[],
            total_documents=0,
        )

    # Use BM25 for demo (doesn't require embeddings)
    if request.strategy in ["bm25", "hybrid"]:
        retriever = BM25Retriever(documents=_demo_documents)
    else:
        # For vector search, would need embeddings
        # For now, fall back to BM25
        retriever = BM25Retriever(documents=_demo_documents)

    result = await retriever.retrieve(
        query=request.query,
        limit=request.limit,
    )

    # Filter by min score
    filtered_results = [
        r for r in result.documents
        if r.score >= request.min_score
    ]

    return SearchResponse(
        query=request.query,
        strategy=request.strategy,
        results=[
            SearchResult(
                document=DocumentResponse(
                    id=r.document.id,
                    content=r.document.content,
                    metadata=r.document.metadata,
                ),
                score=r.score,
                rank=r.rank,
            )
            for r in filtered_results
        ],
        total_documents=len(_demo_documents),
    )


@router.post("/chunk", response_model=ChunkResponse)
async def chunk_text(request: ChunkRequest):
    """
    Chunk text into smaller pieces.

    Strategies:
    - fixed: Fixed-size chunks with overlap
    - semantic: Respect sentence/paragraph boundaries
    - recursive: Try different separators hierarchically
    """
    try:
        chunker = create_chunker(
            strategy=request.strategy,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    chunks = chunker.chunk(request.text)

    return ChunkResponse(
        chunks=[
            {
                "index": c.index,
                "content": c.content,
                "start_char": c.start_char,
                "end_char": c.end_char,
                "metadata": c.metadata,
            }
            for c in chunks
        ],
        total_chunks=len(chunks),
        strategy=request.strategy,
    )


@router.delete("/documents")
async def clear_documents():
    """Clear all documents from the index."""
    global _demo_documents
    count = len(_demo_documents)
    _demo_documents = []
    return {"status": "cleared", "documents_removed": count}
