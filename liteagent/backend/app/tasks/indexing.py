"""
Document indexing tasks.

Handles async document processing:
- Text extraction
- Chunking
- Metadata extraction
"""

from celery import shared_task

from app.celery import app


@app.task(bind=True, queue="indexing")
def index_document(self, document_id: str, options: dict | None = None):
    """
    Index a document asynchronously.

    Args:
        document_id: ID of the document to index
        options: Indexing options (chunk_size, overlap, etc.)

    Returns:
        dict with indexing results
    """
    # Import here to avoid circular imports
    from app.services.document_service import DocumentService

    service = DocumentService()
    result = service.process_document(document_id, options or {})

    return {
        "document_id": document_id,
        "chunks_created": result.get("chunk_count", 0),
        "status": "completed",
    }


@app.task(bind=True, queue="indexing")
def reindex_knowledge_base(self, knowledge_base_id: str):
    """
    Reindex all documents in a knowledge base.

    Args:
        knowledge_base_id: ID of the knowledge base

    Returns:
        dict with reindexing results
    """
    from app.services.knowledge_service import KnowledgeService

    service = KnowledgeService()
    result = service.reindex_all(knowledge_base_id)

    return {
        "knowledge_base_id": knowledge_base_id,
        "documents_processed": result.get("document_count", 0),
        "status": "completed",
    }


@app.task(bind=True, queue="indexing")
def delete_document_chunks(self, document_id: str):
    """
    Delete all chunks for a document.

    Args:
        document_id: ID of the document

    Returns:
        dict with deletion results
    """
    from app.services.document_service import DocumentService

    service = DocumentService()
    result = service.delete_chunks(document_id)

    return {
        "document_id": document_id,
        "chunks_deleted": result.get("deleted_count", 0),
        "status": "completed",
    }
