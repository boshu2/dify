"""
Unit tests for PublishedAppService.

Tests cover publishing, versioning, status management, and webhook operations.
"""

import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime

from models.published_app import (
    PublishedApp,
    PublishedAppStatus,
    PublishedAppVersion,
    PublishedAppVisibility,
    PublishedAppWebhook,
)
from services.published_app_service import (
    PublishedAppService,
    PublishedAppAlreadyExistsError,
    PublishedAppNotFoundError,
)


class TestPublishApp:
    """Tests for app publishing operations."""

    @patch("services.published_app_service.db")
    def test_publish_app_success(self, mock_db):
        """Test successfully publishing an app."""
        mock_app = MagicMock()
        mock_app.id = "app-123"
        mock_app.tenant_id = "tenant-456"

        mock_db.session.scalar.return_value = None  # No existing published app
        mock_db.session.add = MagicMock()
        mock_db.session.commit = MagicMock()

        published_app = PublishedAppService.publish_app(
            app=mock_app,
            name="My Published App",
            created_by="user-789",
            description="Test description",
            visibility=PublishedAppVisibility.PRIVATE,
            default_rate_limit_rpm=100,
        )

        assert published_app.name == "My Published App"
        assert published_app.app_id == "app-123"
        assert published_app.tenant_id == "tenant-456"
        assert published_app.description == "Test description"
        assert published_app.status == PublishedAppStatus.DRAFT.value
        assert published_app.visibility == PublishedAppVisibility.PRIVATE.value
        assert published_app.default_rate_limit_rpm == 100
        assert published_app.slug is not None

    @patch("services.published_app_service.db")
    def test_publish_app_already_exists(self, mock_db):
        """Test publishing an already published app raises error."""
        mock_app = MagicMock()
        mock_app.id = "app-123"
        mock_app.tenant_id = "tenant-456"

        existing_published = MagicMock(spec=PublishedApp)
        mock_db.session.scalar.return_value = existing_published

        with pytest.raises(PublishedAppAlreadyExistsError):
            PublishedAppService.publish_app(
                app=mock_app,
                name="My Published App",
                created_by="user-789",
            )

    def test_generate_slug(self):
        """Test slug generation from name."""
        slug = PublishedApp.generate_slug("My Cool App!")

        assert "my-cool-app" in slug
        assert len(slug) <= 109  # 100 chars name + 1 dash + 8 chars hex


class TestStatusManagement:
    """Tests for published app status management."""

    @patch("services.published_app_service.db")
    def test_go_live(self, mock_db):
        """Test setting a published app to live status."""
        mock_published_app = MagicMock(spec=PublishedApp)
        mock_published_app.id = "pub-123"
        mock_db.session.get.return_value = mock_published_app
        mock_db.session.commit = MagicMock()

        result = PublishedAppService.go_live("pub-123", "user-456")

        assert mock_published_app.status == PublishedAppStatus.PUBLISHED.value
        assert mock_published_app.published_by == "user-456"
        assert mock_published_app.published_at is not None

    @patch("services.published_app_service.db")
    def test_pause(self, mock_db):
        """Test pausing a published app."""
        mock_published_app = MagicMock(spec=PublishedApp)
        mock_db.session.get.return_value = mock_published_app
        mock_db.session.commit = MagicMock()

        result = PublishedAppService.pause("pub-123")

        assert mock_published_app.status == PublishedAppStatus.PAUSED.value

    @patch("services.published_app_service.db")
    def test_deprecate(self, mock_db):
        """Test deprecating a published app."""
        mock_published_app = MagicMock(spec=PublishedApp)
        mock_db.session.get.return_value = mock_published_app
        mock_db.session.commit = MagicMock()

        result = PublishedAppService.deprecate("pub-123")

        assert mock_published_app.status == PublishedAppStatus.DEPRECATED.value

    @patch("services.published_app_service.db")
    def test_archive(self, mock_db):
        """Test archiving a published app."""
        mock_published_app = MagicMock(spec=PublishedApp)
        mock_db.session.get.return_value = mock_published_app
        mock_db.session.commit = MagicMock()

        result = PublishedAppService.archive("pub-123")

        assert mock_published_app.status == PublishedAppStatus.ARCHIVED.value

    @patch("services.published_app_service.db")
    def test_status_not_found(self, mock_db):
        """Test status change on non-existent app raises error."""
        mock_db.session.get.return_value = None

        with pytest.raises(PublishedAppNotFoundError):
            PublishedAppService.go_live("nonexistent", "user-123")


class TestVersionManagement:
    """Tests for version management."""

    @patch("services.published_app_service.db")
    def test_create_version(self, mock_db):
        """Test creating a new version."""
        mock_published_app = MagicMock(spec=PublishedApp)
        mock_published_app.id = "pub-123"
        mock_published_app.version = "1.0.0"
        mock_db.session.add = MagicMock()
        mock_db.session.execute = MagicMock()
        mock_db.session.commit = MagicMock()

        version = PublishedAppService.create_version(
            published_app=mock_published_app,
            version="1.1.0",
            created_by="user-456",
            changelog="Added new feature",
            is_current=True,
        )

        assert version.published_app_id == "pub-123"
        assert version.version == "1.1.0"
        assert version.changelog == "Added new feature"
        assert version.is_current is True
        # Main app version should be updated
        assert mock_published_app.version == "1.1.0"

    @patch("services.published_app_service.db")
    def test_set_current_version(self, mock_db):
        """Test setting a specific version as current."""
        mock_version = MagicMock(spec=PublishedAppVersion)
        mock_version.version = "1.0.0"
        mock_version.changelog = "Initial release"

        mock_published_app = MagicMock(spec=PublishedApp)

        mock_db.session.scalar.return_value = mock_version
        mock_db.session.get.return_value = mock_published_app
        mock_db.session.execute = MagicMock()
        mock_db.session.commit = MagicMock()

        result = PublishedAppService.set_current_version("pub-123", "1.0.0")

        assert mock_version.is_current is True
        assert mock_published_app.version == "1.0.0"

    @patch("services.published_app_service.db")
    def test_set_current_version_not_found(self, mock_db):
        """Test setting non-existent version as current raises error."""
        mock_db.session.scalar.return_value = None

        with pytest.raises(PublishedAppNotFoundError) as exc_info:
            PublishedAppService.set_current_version("pub-123", "99.0.0")

        assert "99.0.0 not found" in str(exc_info.value)

    @patch("services.published_app_service.db")
    def test_deprecate_version(self, mock_db):
        """Test deprecating a specific version."""
        mock_version = MagicMock(spec=PublishedAppVersion)
        mock_db.session.scalar.return_value = mock_version
        mock_db.session.commit = MagicMock()

        result = PublishedAppService.deprecate_version(
            "pub-123", "1.0.0", "Use v2.0.0 instead"
        )

        assert mock_version.is_deprecated is True
        assert mock_version.deprecation_message == "Use v2.0.0 instead"


class TestWebhookManagement:
    """Tests for webhook management."""

    @patch("services.published_app_service.db")
    def test_add_webhook(self, mock_db):
        """Test adding a webhook."""
        mock_db.session.add = MagicMock()
        mock_db.session.commit = MagicMock()

        webhook = PublishedAppService.add_webhook(
            published_app_id="pub-123",
            tenant_id="tenant-456",
            name="My Webhook",
            url="https://example.com/webhook",
            events=["consumer.created", "request.completed"],
            secret="webhook-secret",
        )

        assert webhook.published_app_id == "pub-123"
        assert webhook.name == "My Webhook"
        assert webhook.url == "https://example.com/webhook"
        assert webhook.secret == "webhook-secret"
        assert "consumer.created" in webhook.events

    @patch("services.published_app_service.db")
    def test_delete_webhook(self, mock_db):
        """Test deleting a webhook."""
        mock_webhook = MagicMock(spec=PublishedAppWebhook)
        mock_db.session.get.return_value = mock_webhook
        mock_db.session.delete = MagicMock()
        mock_db.session.commit = MagicMock()

        result = PublishedAppService.delete_webhook("webhook-123")

        assert result is True
        mock_db.session.delete.assert_called_once_with(mock_webhook)

    @patch("services.published_app_service.db")
    def test_delete_webhook_not_found(self, mock_db):
        """Test deleting non-existent webhook returns False."""
        mock_db.session.get.return_value = None

        result = PublishedAppService.delete_webhook("nonexistent")

        assert result is False


class TestAccessControl:
    """Tests for access control helpers."""

    def test_is_accessible_when_live(self):
        """Test is_accessible returns True when app is live."""
        published_app = PublishedApp(
            tenant_id="t1",
            app_id="a1",
            slug="test-app",
            name="Test",
            created_by="u1",
            status=PublishedAppStatus.PUBLISHED.value,
        )

        assert published_app.is_live is True
        assert PublishedAppService.is_accessible(published_app) is True

    def test_is_accessible_when_draft(self):
        """Test is_accessible returns False when app is draft."""
        published_app = PublishedApp(
            tenant_id="t1",
            app_id="a1",
            slug="test-app",
            name="Test",
            created_by="u1",
            status=PublishedAppStatus.DRAFT.value,
        )

        assert published_app.is_live is False
        assert PublishedAppService.is_accessible(published_app) is False

    def test_is_public_when_public_and_live(self):
        """Test is_public returns True when visibility is public and live."""
        published_app = PublishedApp(
            tenant_id="t1",
            app_id="a1",
            slug="test-app",
            name="Test",
            created_by="u1",
            status=PublishedAppStatus.PUBLISHED.value,
            visibility=PublishedAppVisibility.PUBLIC.value,
        )

        assert published_app.is_public is True
        assert PublishedAppService.is_publicly_discoverable(published_app) is True

    def test_is_public_when_unlisted(self):
        """Test is_public returns False when visibility is unlisted."""
        published_app = PublishedApp(
            tenant_id="t1",
            app_id="a1",
            slug="test-app",
            name="Test",
            created_by="u1",
            status=PublishedAppStatus.PUBLISHED.value,
            visibility=PublishedAppVisibility.UNLISTED.value,
        )

        assert published_app.is_public is False


class TestAnalytics:
    """Tests for analytics tracking."""

    @patch("services.published_app_service.db")
    def test_increment_request_count(self, mock_db):
        """Test incrementing request count."""
        mock_published_app = MagicMock(spec=PublishedApp)
        mock_published_app.total_requests = 100
        mock_db.session.get.return_value = mock_published_app
        mock_db.session.commit = MagicMock()

        PublishedAppService.increment_request_count("pub-123")

        assert mock_published_app.total_requests == 101

    @patch("services.published_app_service.db")
    def test_increment_consumer_count(self, mock_db):
        """Test incrementing consumer count."""
        mock_published_app = MagicMock(spec=PublishedApp)
        mock_published_app.total_consumers = 50
        mock_db.session.get.return_value = mock_published_app
        mock_db.session.commit = MagicMock()

        PublishedAppService.increment_consumer_count("pub-123")

        assert mock_published_app.total_consumers == 51


class TestUnpublish:
    """Tests for unpublishing apps."""

    @patch("services.published_app_service.db")
    def test_unpublish_success(self, mock_db):
        """Test successfully unpublishing an app."""
        mock_published_app = MagicMock(spec=PublishedApp)
        mock_db.session.get.return_value = mock_published_app
        mock_db.session.execute = MagicMock()
        mock_db.session.delete = MagicMock()
        mock_db.session.commit = MagicMock()

        result = PublishedAppService.unpublish("pub-123")

        assert result is True
        mock_db.session.delete.assert_called_once_with(mock_published_app)

    @patch("services.published_app_service.db")
    def test_unpublish_not_found(self, mock_db):
        """Test unpublishing non-existent app returns False."""
        mock_db.session.get.return_value = None

        result = PublishedAppService.unpublish("nonexistent")

        assert result is False


class TestGetters:
    """Tests for getter methods."""

    @patch("services.published_app_service.db")
    def test_get_published_app_by_slug(self, mock_db):
        """Test getting published app by slug."""
        mock_app = MagicMock(spec=PublishedApp)
        mock_app.slug = "my-cool-app"
        mock_db.session.scalar.return_value = mock_app

        result = PublishedAppService.get_published_app_by_slug("my-cool-app")

        assert result == mock_app
        assert result.slug == "my-cool-app"

    @patch("services.published_app_service.db")
    def test_get_published_app_by_app_id(self, mock_db):
        """Test getting published app by internal app ID."""
        mock_app = MagicMock(spec=PublishedApp)
        mock_app.app_id = "app-123"
        mock_db.session.scalar.return_value = mock_app

        result = PublishedAppService.get_published_app_by_app_id(
            "app-123", "tenant-456"
        )

        assert result == mock_app

    def test_api_base_url(self):
        """Test API base URL generation."""
        with patch("configs.dify_config") as mock_config:
            mock_config.SERVICE_API_URL = "https://api.example.com"

            published_app = PublishedApp(
                tenant_id="t1",
                app_id="a1",
                slug="test-app-abc123",
                name="Test",
                created_by="u1",
            )

            assert published_app.api_base_url == "https://api.example.com/published/test-app-abc123"
