"""
Celery tasks for webhook delivery.
Handles async webhook notifications with retry.
"""
import httpx
from celery import shared_task
from celery.utils.log import get_task_logger

logger = get_task_logger(__name__)


@shared_task(
    bind=True,
    max_retries=5,
    default_retry_delay=30,
    autoretry_for=(httpx.RequestError,),
    retry_backoff=True,
    retry_backoff_max=600,  # Max 10 minutes
)
def deliver_webhook(
    self,
    webhook_url: str,
    event_type: str,
    payload: dict,
    headers: dict | None = None,
) -> dict:
    """
    Deliver a webhook notification.

    Args:
        webhook_url: URL to send the webhook to.
        event_type: Type of event (e.g., 'agent.chat.completed').
        payload: Event payload data.
        headers: Optional additional headers.

    Returns:
        Dict with delivery status.
    """
    logger.info(f"Delivering webhook to {webhook_url}: {event_type}")

    # Get delivery ID from Celery request if available
    delivery_id = getattr(self.request, "id", None) or "unknown"

    default_headers = {
        "Content-Type": "application/json",
        "X-LiteAgent-Event": event_type,
        "X-LiteAgent-Delivery-ID": delivery_id,
    }

    if headers:
        default_headers.update(headers)

    try:
        with httpx.Client(timeout=30) as client:
            response = client.post(
                webhook_url,
                json=payload,
                headers=default_headers,
            )
            response.raise_for_status()

        return {
            "status": "delivered",
            "status_code": response.status_code,
            "webhook_url": webhook_url,
            "event_type": event_type,
        }

    except httpx.HTTPStatusError as exc:
        logger.warning(
            f"Webhook returned error status: {exc.response.status_code}"
        )
        if exc.response.status_code >= 500:
            # Retry on server errors
            raise self.retry(exc=exc)
        # Don't retry on client errors (4xx)
        return {
            "status": "failed",
            "status_code": exc.response.status_code,
            "error": str(exc),
        }

    except httpx.RequestError as exc:
        logger.error(f"Webhook delivery failed: {exc}")
        raise


@shared_task
def cleanup_failed_webhooks(max_age_hours: int = 24) -> int:
    """
    Clean up old failed webhook delivery records.

    Args:
        max_age_hours: Max age of records to keep.

    Returns:
        Number of records cleaned up.
    """
    logger.info(f"Cleaning up failed webhooks older than {max_age_hours} hours")
    return 0
