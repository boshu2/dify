"""
Celery application configuration for background jobs.
Uses Redis as broker and result backend.
"""
import os
from celery import Celery
from kombu import Queue

# Celery configuration
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

# Create Celery app
celery_app = Celery(
    "liteagent",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=[
        "app.tasks.datasource_tasks",
        "app.tasks.llm_tasks",
        "app.tasks.webhook_tasks",
    ],
)

# Celery configuration
celery_app.conf.update(
    # Task settings
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,

    # Task execution
    task_acks_late=True,  # Acknowledge after task completes
    task_reject_on_worker_lost=True,
    task_time_limit=300,  # 5 minute hard limit
    task_soft_time_limit=240,  # 4 minute soft limit

    # Retry settings
    task_default_retry_delay=60,  # 1 minute
    task_max_retries=3,

    # Queue configuration
    task_queues=(
        Queue("default", routing_key="default"),
        Queue("high_priority", routing_key="high"),
        Queue("low_priority", routing_key="low"),
    ),
    task_default_queue="default",
    task_default_routing_key="default",

    # Result backend settings
    result_expires=3600,  # 1 hour

    # Worker settings
    worker_prefetch_multiplier=1,  # Fair task distribution
    worker_concurrency=4,
)


# Task routing
celery_app.conf.task_routes = {
    "app.tasks.llm_tasks.*": {"queue": "default"},
    "app.tasks.datasource_tasks.refresh_*": {"queue": "low_priority"},
    "app.tasks.webhook_tasks.*": {"queue": "high_priority"},
}


def get_celery_app() -> Celery:
    """Get the configured Celery application."""
    return celery_app
