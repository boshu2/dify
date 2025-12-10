"""
Embedding generation tasks.

Handles async embedding generation:
- Document chunk embeddings
- Query embeddings
- Batch processing
"""

from celery import shared_task

from app.celery import app


@app.task(bind=True, queue="embedding")
def generate_embeddings(self, chunk_ids: list[str], model: str | None = None):
    """
    Generate embeddings for document chunks.

    Args:
        chunk_ids: List of chunk IDs to embed
        model: Embedding model to use (optional)

    Returns:
        dict with embedding results
    """
    from app.services.embedding_service import EmbeddingService

    service = EmbeddingService()
    result = service.embed_chunks(chunk_ids, model)

    return {
        "chunks_processed": len(chunk_ids),
        "embeddings_created": result.get("count", 0),
        "status": "completed",
    }


@app.task(bind=True, queue="embedding")
def batch_embed_documents(self, document_ids: list[str], model: str | None = None):
    """
    Generate embeddings for multiple documents.

    Args:
        document_ids: List of document IDs
        model: Embedding model to use

    Returns:
        dict with batch results
    """
    from app.services.embedding_service import EmbeddingService

    service = EmbeddingService()
    results = []

    for doc_id in document_ids:
        try:
            result = service.embed_document(doc_id, model)
            results.append({"document_id": doc_id, "status": "completed"})
        except Exception as e:
            results.append({"document_id": doc_id, "status": "failed", "error": str(e)})

    return {
        "documents_processed": len(document_ids),
        "results": results,
        "status": "completed",
    }


@app.task(bind=True, queue="embedding")
def update_embedding_model(self, knowledge_base_id: str, new_model: str):
    """
    Re-embed all documents with a new model.

    Args:
        knowledge_base_id: Knowledge base ID
        new_model: New embedding model to use

    Returns:
        dict with update results
    """
    from app.services.embedding_service import EmbeddingService

    service = EmbeddingService()
    result = service.reembed_knowledge_base(knowledge_base_id, new_model)

    return {
        "knowledge_base_id": knowledge_base_id,
        "documents_updated": result.get("count", 0),
        "new_model": new_model,
        "status": "completed",
    }
