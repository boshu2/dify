"""
Celery application configuration.

Provides async task processing for:
- Document indexing
- Embedding generation
- Workflow execution
- Background jobs
"""

import os

from celery import Celery

# Get broker URL from environment
CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/1")
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/1")

# Create Celery app
app = Celery(
    "liteagent",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
    include=[
        "app.tasks.indexing",
        "app.tasks.embedding",
        "app.tasks.workflow",
    ],
)

# Celery configuration
app.conf.update(
    # Task settings
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,

    # Task routing
    task_routes={
        "app.tasks.indexing.*": {"queue": "indexing"},
        "app.tasks.embedding.*": {"queue": "embedding"},
        "app.tasks.workflow.*": {"queue": "workflow"},
    },

    # Task execution settings
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    task_time_limit=3600,  # 1 hour hard limit
    task_soft_time_limit=3000,  # 50 minute soft limit

    # Worker settings
    worker_prefetch_multiplier=1,  # Disable prefetching for long tasks
    worker_max_tasks_per_child=100,  # Recycle workers after 100 tasks

    # Result settings
    result_expires=86400,  # Results expire after 1 day

    # Beat scheduler
    beat_schedule={
        "cleanup-expired-sessions": {
            "task": "app.tasks.maintenance.cleanup_sessions",
            "schedule": 3600.0,  # Every hour
        },
        "cleanup-orphaned-files": {
            "task": "app.tasks.maintenance.cleanup_files",
            "schedule": 86400.0,  # Daily
        },
    },
)

# Autodiscover tasks
app.autodiscover_tasks(["app.tasks"])


@app.task(bind=True)
def debug_task(self):
    """Debug task for testing Celery connectivity."""
    print(f"Request: {self.request!r}")
    return {"status": "ok", "task_id": self.request.id}
