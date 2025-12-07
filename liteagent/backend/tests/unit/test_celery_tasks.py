"""
Unit tests for Celery background tasks.
Tests task logic without requiring actual Celery broker.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import httpx

from app.tasks.datasource_tasks import (
    refresh_datasource,
    refresh_all_datasources,
    cleanup_expired_datasources,
)
from app.tasks.llm_tasks import (
    process_chat_async,
    batch_embed_documents,
    summarize_conversation,
)
from app.tasks.webhook_tasks import (
    deliver_webhook,
    cleanup_failed_webhooks,
)


class TestRefreshDatasource:
    """Tests for datasource refresh task."""

    def test_refresh_datasource_success(self):
        """Test successful datasource refresh."""
        # Create mock task with request
        task = refresh_datasource

        # Call the underlying function directly
        result = refresh_datasource.run("ds-123")

        assert result["datasource_id"] == "ds-123"
        assert result["status"] == "refreshed"

    def test_refresh_datasource_returns_dict(self):
        """Test refresh returns proper dict structure."""
        result = refresh_datasource.run("test-id")

        assert isinstance(result, dict)
        assert "datasource_id" in result
        assert "status" in result


class TestRefreshAllDatasources:
    """Tests for bulk datasource refresh task."""

    def test_refresh_all_returns_counts(self):
        """Test bulk refresh returns refresh counts."""
        result = refresh_all_datasources.run()

        assert "refreshed" in result
        assert "failed" in result
        assert isinstance(result["refreshed"], int)
        assert isinstance(result["failed"], int)


class TestCleanupExpiredDatasources:
    """Tests for datasource cleanup task."""

    def test_cleanup_returns_count(self):
        """Test cleanup returns number cleaned."""
        result = cleanup_expired_datasources.run()

        assert isinstance(result, int)
        assert result >= 0


class TestProcessChatAsync:
    """Tests for async chat processing task."""

    def test_process_chat_success(self):
        """Test successful chat processing."""
        messages = [
            {"role": "user", "content": "Hello!"},
        ]

        result = process_chat_async.run("agent-123", messages)

        assert result["agent_id"] == "agent-123"
        assert result["status"] == "completed"
        assert "response" in result

    def test_process_chat_with_user_id(self):
        """Test chat processing with user ID."""
        messages = [{"role": "user", "content": "Test"}]

        result = process_chat_async.run(
            "agent-123",
            messages,
            user_id="user-456",
        )

        assert result["agent_id"] == "agent-123"

    def test_process_chat_returns_dict(self):
        """Test chat returns proper structure."""
        result = process_chat_async.run("agent-1", [])

        assert isinstance(result, dict)


class TestBatchEmbedDocuments:
    """Tests for batch embedding task."""

    def test_batch_embed_empty_list(self):
        """Test embedding empty document list."""
        result = batch_embed_documents.run([])

        assert isinstance(result, list)
        assert len(result) == 0

    def test_batch_embed_documents(self):
        """Test embedding multiple documents."""
        docs = ["Hello world", "Test document"]

        result = batch_embed_documents.run(docs)

        # Currently returns empty, but tests the interface
        assert isinstance(result, list)

    def test_batch_embed_with_model(self):
        """Test embedding with specific model."""
        result = batch_embed_documents.run(
            ["test"],
            model="text-embedding-ada-002",
        )

        assert isinstance(result, list)


class TestSummarizeConversation:
    """Tests for conversation summarization task."""

    def test_summarize_returns_string(self):
        """Test summarization returns string."""
        result = summarize_conversation.run("conv-123")

        assert isinstance(result, str)

    def test_summarize_with_max_tokens(self):
        """Test summarization with token limit."""
        result = summarize_conversation.run(
            "conv-123",
            max_tokens=1000,
        )

        assert isinstance(result, str)


class TestDeliverWebhook:
    """Tests for webhook delivery task."""

    def test_deliver_webhook_task_exists(self):
        """Test webhook delivery task is properly defined."""
        assert deliver_webhook is not None
        assert hasattr(deliver_webhook, "run")
        assert hasattr(deliver_webhook, "max_retries")

    def test_deliver_webhook_task_config(self):
        """Test webhook task has proper retry configuration."""
        assert deliver_webhook.max_retries == 5
        assert deliver_webhook.retry_backoff is True
        assert deliver_webhook.retry_backoff_max == 600

    def test_deliver_webhook_default_retry_delay(self):
        """Test webhook default retry delay."""
        assert deliver_webhook.default_retry_delay == 30

    @patch("app.tasks.webhook_tasks.httpx.Client")
    def test_webhook_calls_client_post(self, mock_client_class):
        """Test webhook uses httpx client to POST."""
        mock_client = MagicMock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_client.post.return_value = mock_response
        mock_client_class.return_value.__enter__ = Mock(return_value=mock_client)
        mock_client_class.return_value.__exit__ = Mock(return_value=None)

        # Call the task's run method
        result = deliver_webhook.run(
            "https://example.com/webhook",
            "test.event",
            {"data": "test"},
        )

        # Verify httpx Client was used
        mock_client_class.assert_called_once_with(timeout=30)

        # Verify result structure
        assert result["status"] == "delivered"
        assert result["status_code"] == 200
        assert result["event_type"] == "test.event"

    @patch("app.tasks.webhook_tasks.httpx.Client")
    def test_webhook_with_custom_headers(self, mock_client_class):
        """Test webhook includes custom headers."""
        mock_client = MagicMock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_client.post.return_value = mock_response
        mock_client_class.return_value.__enter__ = Mock(return_value=mock_client)
        mock_client_class.return_value.__exit__ = Mock(return_value=None)

        result = deliver_webhook.run(
            "https://example.com/webhook",
            "test.event",
            {"data": "test"},
            headers={"X-Custom": "value"},
        )

        # Verify custom headers were passed
        call_kwargs = mock_client.post.call_args[1]
        assert "X-Custom" in call_kwargs["headers"]
        assert call_kwargs["headers"]["X-Custom"] == "value"


class TestCleanupFailedWebhooks:
    """Tests for webhook cleanup task."""

    def test_cleanup_returns_count(self):
        """Test cleanup returns number of records cleaned."""
        result = cleanup_failed_webhooks.run()

        assert isinstance(result, int)
        assert result >= 0

    def test_cleanup_with_max_age(self):
        """Test cleanup with custom max age."""
        result = cleanup_failed_webhooks.run(max_age_hours=48)

        assert isinstance(result, int)


class TestCeleryAppConfiguration:
    """Tests for Celery app configuration."""

    def test_celery_app_exists(self):
        """Test Celery app is properly configured."""
        from app.core.celery_app import celery_app

        assert celery_app is not None
        assert celery_app.main == "liteagent"

    def test_celery_app_has_queues(self):
        """Test Celery app has queue configuration."""
        from app.core.celery_app import celery_app

        queues = celery_app.conf.task_queues
        queue_names = [q.name for q in queues]

        assert "default" in queue_names
        assert "high_priority" in queue_names
        assert "low_priority" in queue_names

    def test_celery_task_routing(self):
        """Test task routing configuration."""
        from app.core.celery_app import celery_app

        routes = celery_app.conf.task_routes

        assert "app.tasks.llm_tasks.*" in routes
        assert "app.tasks.webhook_tasks.*" in routes

    def test_celery_serialization_settings(self):
        """Test serialization is JSON."""
        from app.core.celery_app import celery_app

        assert celery_app.conf.task_serializer == "json"
        assert celery_app.conf.result_serializer == "json"

    def test_get_celery_app(self):
        """Test get_celery_app returns the app."""
        from app.core.celery_app import get_celery_app, celery_app

        app = get_celery_app()
        assert app is celery_app


class TestTaskRetryConfiguration:
    """Tests for task retry settings."""

    def test_datasource_task_has_retry(self):
        """Test datasource task has retry configuration."""
        assert refresh_datasource.max_retries == 3
        assert refresh_datasource.default_retry_delay == 60

    def test_webhook_task_has_backoff(self):
        """Test webhook task has exponential backoff."""
        assert deliver_webhook.max_retries == 5
        assert deliver_webhook.retry_backoff is True
        assert deliver_webhook.retry_backoff_max == 600

    def test_chat_task_has_retry(self):
        """Test chat task has retry configuration."""
        assert process_chat_async.max_retries == 3
