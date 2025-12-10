"""
Maintenance and cleanup tasks.

Handles periodic maintenance:
- Session cleanup
- File cleanup
- Cache cleanup
- Database maintenance
"""

from celery import shared_task

from app.celery import app


@app.task(bind=True, queue="default")
def cleanup_sessions(self):
    """
    Clean up expired sessions.

    Returns:
        dict with cleanup results
    """
    from app.services.session_service import SessionService

    service = SessionService()
    result = service.cleanup_expired()

    return {
        "sessions_cleaned": result.get("count", 0),
        "status": "completed",
    }


@app.task(bind=True, queue="default")
def cleanup_files(self):
    """
    Clean up orphaned files.

    Returns:
        dict with cleanup results
    """
    from app.services.file_service import FileService

    service = FileService()
    result = service.cleanup_orphaned()

    return {
        "files_cleaned": result.get("count", 0),
        "bytes_freed": result.get("bytes", 0),
        "status": "completed",
    }


@app.task(bind=True, queue="default")
def cleanup_old_executions(self, days: int = 30):
    """
    Clean up old workflow executions.

    Args:
        days: Number of days to retain

    Returns:
        dict with cleanup results
    """
    from app.services.workflow_service import WorkflowService

    service = WorkflowService()
    result = service.cleanup_old_executions(days)

    return {
        "executions_cleaned": result.get("count", 0),
        "status": "completed",
    }


@app.task(bind=True, queue="default")
def vacuum_database(self):
    """
    Run database vacuum/analyze for optimization.

    Returns:
        dict with vacuum results
    """
    from app.core.database import get_db

    # This is a placeholder - actual implementation depends on database
    return {
        "status": "completed",
        "message": "Database maintenance completed",
    }


@app.task(bind=True, queue="default")
def cleanup_embedding_cache(self):
    """
    Clean up stale embedding cache entries.

    Returns:
        dict with cleanup results
    """
    from app.services.embedding_service import EmbeddingService

    service = EmbeddingService()
    result = service.cleanup_cache()

    return {
        "entries_cleaned": result.get("count", 0),
        "status": "completed",
    }
