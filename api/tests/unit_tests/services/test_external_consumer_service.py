"""
Unit tests for ExternalConsumerService.

Tests cover consumer CRUD, API key authentication, app access management,
and usage tracking.
"""

import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta

from models.external_consumer import (
    ExternalConsumer,
    ExternalConsumerAppAccess,
    ExternalConsumerAuthType,
    ExternalConsumerStatus,
    ExternalConsumerUsageLog,
)
from services.external_consumer_service import (
    ExternalConsumerService,
    ExternalConsumerAccessDeniedError,
    ExternalConsumerNotFoundError,
    ExternalConsumerQuotaExceededError,
)


class TestExternalConsumerCRUD:
    """Tests for consumer CRUD operations."""

    @patch("services.external_consumer_service.db")
    def test_create_consumer_with_api_key(self, mock_db):
        """Test creating a consumer with API key authentication."""
        mock_db.session.add = MagicMock()
        mock_db.session.commit = MagicMock()

        consumer, api_key = ExternalConsumerService.create_consumer(
            tenant_id="tenant-123",
            name="Test Consumer",
            email="test@example.com",
            created_by="user-123",
            organization="Test Org",
            auth_type=ExternalConsumerAuthType.API_KEY,
            rate_limit_rpm=100,
            quota_total=10000,
        )

        assert consumer.name == "Test Consumer"
        assert consumer.email == "test@example.com"
        assert consumer.tenant_id == "tenant-123"
        assert consumer.organization == "Test Org"
        assert consumer.rate_limit_rpm == 100
        assert consumer.quota_total == 10000
        assert api_key is not None
        assert api_key.startswith("ec_")
        assert consumer.api_key_hash is not None
        assert consumer.api_key_prefix is not None
        mock_db.session.add.assert_called_once()
        mock_db.session.commit.assert_called_once()

    @patch("services.external_consumer_service.db")
    def test_create_consumer_with_oauth(self, mock_db):
        """Test creating a consumer with OAuth authentication."""
        mock_db.session.add = MagicMock()
        mock_db.session.commit = MagicMock()

        consumer, api_key = ExternalConsumerService.create_consumer(
            tenant_id="tenant-123",
            name="OAuth Consumer",
            email="oauth@example.com",
            created_by="user-123",
            auth_type=ExternalConsumerAuthType.OAUTH,
        )

        assert consumer.auth_type == ExternalConsumerAuthType.OAUTH.value
        assert api_key is None  # No API key for OAuth
        assert consumer.api_key_hash is None

    @patch("services.external_consumer_service.db")
    def test_get_consumer(self, mock_db):
        """Test getting a consumer by ID."""
        mock_consumer = MagicMock(spec=ExternalConsumer)
        mock_consumer.id = "consumer-123"
        mock_db.session.get.return_value = mock_consumer

        result = ExternalConsumerService.get_consumer("consumer-123")

        assert result == mock_consumer
        mock_db.session.get.assert_called_once_with(ExternalConsumer, "consumer-123")

    @patch("services.external_consumer_service.db")
    def test_update_consumer(self, mock_db):
        """Test updating consumer details."""
        mock_consumer = MagicMock(spec=ExternalConsumer)
        mock_consumer.id = "consumer-123"
        mock_db.session.get.return_value = mock_consumer
        mock_db.session.commit = MagicMock()

        result = ExternalConsumerService.update_consumer(
            consumer_id="consumer-123",
            name="Updated Name",
            rate_limit_rpm=200,
        )

        assert mock_consumer.name == "Updated Name"
        assert mock_consumer.rate_limit_rpm == 200
        mock_db.session.commit.assert_called_once()

    @patch("services.external_consumer_service.db")
    def test_update_consumer_not_found(self, mock_db):
        """Test updating a non-existent consumer raises error."""
        mock_db.session.get.return_value = None

        with pytest.raises(ExternalConsumerNotFoundError):
            ExternalConsumerService.update_consumer(
                consumer_id="nonexistent",
                name="Updated Name",
            )


class TestConsumerStatus:
    """Tests for consumer status management."""

    @patch("services.external_consumer_service.db")
    def test_suspend_consumer(self, mock_db):
        """Test suspending a consumer."""
        mock_consumer = MagicMock(spec=ExternalConsumer)
        mock_db.session.get.return_value = mock_consumer
        mock_db.session.commit = MagicMock()

        result = ExternalConsumerService.suspend_consumer("consumer-123")

        assert mock_consumer.status == ExternalConsumerStatus.SUSPENDED.value

    @patch("services.external_consumer_service.db")
    def test_activate_consumer(self, mock_db):
        """Test activating a consumer."""
        mock_consumer = MagicMock(spec=ExternalConsumer)
        mock_consumer.status = ExternalConsumerStatus.SUSPENDED.value
        mock_db.session.get.return_value = mock_consumer
        mock_db.session.commit = MagicMock()

        result = ExternalConsumerService.activate_consumer("consumer-123")

        assert mock_consumer.status == ExternalConsumerStatus.ACTIVE.value

    @patch("services.external_consumer_service.db")
    def test_revoke_consumer(self, mock_db):
        """Test revoking a consumer."""
        mock_consumer = MagicMock(spec=ExternalConsumer)
        mock_db.session.get.return_value = mock_consumer
        mock_db.session.commit = MagicMock()

        result = ExternalConsumerService.revoke_consumer("consumer-123")

        assert mock_consumer.status == ExternalConsumerStatus.REVOKED.value


class TestAPIKeyAuthentication:
    """Tests for API key authentication."""

    def test_api_key_generation(self):
        """Test API key generation produces valid format."""
        key, prefix, key_hash = ExternalConsumer.generate_api_key()

        assert key.startswith("ec_")
        assert len(key) > 40  # Base64 encoded 32 bytes + prefix
        assert prefix == key[:12]
        assert len(key_hash) == 64  # SHA256 hex digest

    def test_api_key_verification_success(self):
        """Test API key verification with correct key."""
        key, prefix, key_hash = ExternalConsumer.generate_api_key()

        result = ExternalConsumer.verify_api_key(key, key_hash)

        assert result is True

    def test_api_key_verification_failure(self):
        """Test API key verification with wrong key."""
        key, prefix, key_hash = ExternalConsumer.generate_api_key()
        wrong_key = "ec_wrong_key_here"

        result = ExternalConsumer.verify_api_key(wrong_key, key_hash)

        assert result is False

    @patch("services.external_consumer_service.db")
    def test_authenticate_by_api_key_success(self, mock_db):
        """Test successful API key authentication."""
        # Generate a real key for testing
        key, prefix, key_hash = ExternalConsumer.generate_api_key()

        mock_consumer = MagicMock(spec=ExternalConsumer)
        mock_consumer.api_key_hash = key_hash
        mock_consumer.is_active = True
        mock_consumer.status = ExternalConsumerStatus.ACTIVE.value
        mock_db.session.scalar.return_value = mock_consumer
        mock_db.session.commit = MagicMock()

        result = ExternalConsumerService.authenticate_by_api_key("tenant-123", key)

        assert result == mock_consumer
        assert mock_consumer.last_active_at is not None

    @patch("services.external_consumer_service.db")
    def test_authenticate_by_api_key_invalid(self, mock_db):
        """Test authentication with invalid API key."""
        mock_db.session.scalar.return_value = None

        with pytest.raises(ExternalConsumerAccessDeniedError) as exc_info:
            ExternalConsumerService.authenticate_by_api_key("tenant-123", "invalid_key")

        assert "Invalid API key" in str(exc_info.value)

    @patch("services.external_consumer_service.db")
    def test_authenticate_suspended_consumer(self, mock_db):
        """Test authentication fails for suspended consumer."""
        key, prefix, key_hash = ExternalConsumer.generate_api_key()

        mock_consumer = MagicMock(spec=ExternalConsumer)
        mock_consumer.api_key_hash = key_hash
        mock_consumer.is_active = False
        mock_consumer.status = ExternalConsumerStatus.SUSPENDED.value
        mock_db.session.scalar.return_value = mock_consumer

        with pytest.raises(ExternalConsumerAccessDeniedError) as exc_info:
            ExternalConsumerService.authenticate_by_api_key("tenant-123", key)

        assert "suspended" in str(exc_info.value).lower()


class TestAppAccess:
    """Tests for app access management."""

    @patch("services.external_consumer_service.db")
    def test_grant_app_access(self, mock_db):
        """Test granting app access to a consumer."""
        mock_db.session.scalar.return_value = None  # No existing access
        mock_db.session.add = MagicMock()
        mock_db.session.commit = MagicMock()

        access = ExternalConsumerService.grant_app_access(
            consumer_id="consumer-123",
            app_id="app-456",
            tenant_id="tenant-789",
            granted_by="user-123",
            can_invoke=True,
            can_view_logs=True,
            custom_rate_limit_rpm=50,
            allowed_scopes=["app:invoke", "conversation:read"],
        )

        assert access.consumer_id == "consumer-123"
        assert access.app_id == "app-456"
        assert access.can_invoke is True
        assert access.can_view_logs is True
        assert access.custom_rate_limit_rpm == 50
        mock_db.session.add.assert_called_once()

    @patch("services.external_consumer_service.db")
    def test_revoke_app_access(self, mock_db):
        """Test revoking app access from a consumer."""
        mock_access = MagicMock(spec=ExternalConsumerAppAccess)
        mock_db.session.scalar.return_value = mock_access
        mock_db.session.delete = MagicMock()
        mock_db.session.commit = MagicMock()

        result = ExternalConsumerService.revoke_app_access("consumer-123", "app-456")

        assert result is True
        mock_db.session.delete.assert_called_once_with(mock_access)

    @patch("services.external_consumer_service.db")
    def test_check_app_access_success(self, mock_db):
        """Test checking app access succeeds with valid access."""
        mock_consumer = MagicMock(spec=ExternalConsumer)
        mock_consumer.id = "consumer-123"
        mock_consumer.is_within_quota = True

        mock_access = MagicMock(spec=ExternalConsumerAppAccess)
        mock_access.is_valid = True
        mock_access.can_invoke = True
        mock_access.is_within_quota = True
        mock_access.allowed_scopes = None

        mock_db.session.scalar.return_value = mock_access

        result = ExternalConsumerService.check_app_access(mock_consumer, "app-456")

        assert result == mock_access

    @patch("services.external_consumer_service.db")
    def test_check_app_access_no_access(self, mock_db):
        """Test checking app access fails without access grant."""
        mock_consumer = MagicMock(spec=ExternalConsumer)
        mock_consumer.id = "consumer-123"
        mock_db.session.scalar.return_value = None

        with pytest.raises(ExternalConsumerAccessDeniedError) as exc_info:
            ExternalConsumerService.check_app_access(mock_consumer, "app-456")

        assert "does not have access" in str(exc_info.value)

    @patch("services.external_consumer_service.db")
    def test_check_app_access_expired(self, mock_db):
        """Test checking app access fails with expired access."""
        mock_consumer = MagicMock(spec=ExternalConsumer)
        mock_consumer.id = "consumer-123"

        mock_access = MagicMock(spec=ExternalConsumerAppAccess)
        mock_access.is_valid = False

        mock_db.session.scalar.return_value = mock_access

        with pytest.raises(ExternalConsumerAccessDeniedError) as exc_info:
            ExternalConsumerService.check_app_access(mock_consumer, "app-456")

        assert "expired" in str(exc_info.value).lower()

    @patch("services.external_consumer_service.db")
    def test_check_app_access_quota_exceeded(self, mock_db):
        """Test checking app access fails when quota exceeded."""
        mock_consumer = MagicMock(spec=ExternalConsumer)
        mock_consumer.id = "consumer-123"
        mock_consumer.is_within_quota = False

        mock_access = MagicMock(spec=ExternalConsumerAppAccess)
        mock_access.is_valid = True
        mock_access.can_invoke = True

        mock_db.session.scalar.return_value = mock_access

        with pytest.raises(ExternalConsumerQuotaExceededError):
            ExternalConsumerService.check_app_access(mock_consumer, "app-456")


class TestUsageTracking:
    """Tests for usage tracking."""

    @patch("services.external_consumer_service.db")
    def test_log_usage(self, mock_db):
        """Test logging API usage."""
        mock_consumer = MagicMock(spec=ExternalConsumer)
        mock_consumer.quota_used = 5
        mock_db.session.get.return_value = mock_consumer

        mock_access = MagicMock(spec=ExternalConsumerAppAccess)
        mock_access.app_quota_used = 10
        mock_db.session.scalar.return_value = mock_access

        mock_db.session.add = MagicMock()
        mock_db.session.commit = MagicMock()

        log = ExternalConsumerService.log_usage(
            consumer_id="consumer-123",
            app_id="app-456",
            tenant_id="tenant-789",
            endpoint="/v1/chat",
            method="POST",
            status_code=200,
            response_time_ms=150,
            prompt_tokens=100,
            completion_tokens=50,
            estimated_cost=0.001,
        )

        assert log.consumer_id == "consumer-123"
        assert log.app_id == "app-456"
        assert log.endpoint == "/v1/chat"
        assert log.status_code == 200
        assert log.total_tokens == 150
        assert mock_consumer.quota_used == 6  # Incremented
        assert mock_access.app_quota_used == 11  # Incremented

    @patch("services.external_consumer_service.db")
    def test_reset_consumer_quota(self, mock_db):
        """Test resetting consumer quota."""
        mock_consumer = MagicMock(spec=ExternalConsumer)
        mock_consumer.quota_used = 100
        mock_db.session.get.return_value = mock_consumer
        mock_db.session.commit = MagicMock()

        result = ExternalConsumerService.reset_consumer_quota("consumer-123")

        assert mock_consumer.quota_used == 0
        assert mock_consumer.quota_reset_at is not None


class TestAccessValidity:
    """Tests for access validity checks."""

    def test_access_is_valid_no_dates(self):
        """Test access is valid when no date restrictions."""
        access = ExternalConsumerAppAccess(
            consumer_id="c1",
            app_id="a1",
            tenant_id="t1",
            granted_by="u1",
            valid_from=None,
            valid_until=None,
        )

        assert access.is_valid is True

    def test_access_is_valid_within_dates(self):
        """Test access is valid when within date range."""
        access = ExternalConsumerAppAccess(
            consumer_id="c1",
            app_id="a1",
            tenant_id="t1",
            granted_by="u1",
            valid_from=datetime.utcnow() - timedelta(days=1),
            valid_until=datetime.utcnow() + timedelta(days=1),
        )

        assert access.is_valid is True

    def test_access_not_yet_valid(self):
        """Test access is invalid when before valid_from."""
        access = ExternalConsumerAppAccess(
            consumer_id="c1",
            app_id="a1",
            tenant_id="t1",
            granted_by="u1",
            valid_from=datetime.utcnow() + timedelta(days=1),
            valid_until=None,
        )

        assert access.is_valid is False

    def test_access_expired(self):
        """Test access is invalid when after valid_until."""
        access = ExternalConsumerAppAccess(
            consumer_id="c1",
            app_id="a1",
            tenant_id="t1",
            granted_by="u1",
            valid_from=None,
            valid_until=datetime.utcnow() - timedelta(days=1),
        )

        assert access.is_valid is False


class TestQuotaChecks:
    """Tests for quota checking."""

    def test_consumer_within_quota(self):
        """Test consumer is within quota."""
        consumer = ExternalConsumer(
            tenant_id="t1",
            name="Test",
            email="test@test.com",
            created_by="u1",
            quota_total=100,
            quota_used=50,
        )

        assert consumer.is_within_quota is True

    def test_consumer_exceeded_quota(self):
        """Test consumer has exceeded quota."""
        consumer = ExternalConsumer(
            tenant_id="t1",
            name="Test",
            email="test@test.com",
            created_by="u1",
            quota_total=100,
            quota_used=100,
        )

        assert consumer.is_within_quota is False

    def test_consumer_no_quota_limit(self):
        """Test consumer with no quota limit is always within quota."""
        consumer = ExternalConsumer(
            tenant_id="t1",
            name="Test",
            email="test@test.com",
            created_by="u1",
            quota_total=None,
            quota_used=1000000,
        )

        assert consumer.is_within_quota is True

    def test_app_access_within_quota(self):
        """Test app access is within quota."""
        access = ExternalConsumerAppAccess(
            consumer_id="c1",
            app_id="a1",
            tenant_id="t1",
            granted_by="u1",
            app_quota_total=100,
            app_quota_used=50,
        )

        assert access.is_within_quota is True

    def test_app_access_no_quota_limit(self):
        """Test app access with no quota limit is always within quota."""
        access = ExternalConsumerAppAccess(
            consumer_id="c1",
            app_id="a1",
            tenant_id="t1",
            granted_by="u1",
            app_quota_total=None,
            app_quota_used=1000000,
        )

        assert access.is_within_quota is True
