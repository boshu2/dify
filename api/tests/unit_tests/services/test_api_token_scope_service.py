"""
Unit tests for ApiTokenScopeService.

Tests cover scope management, token validation, and configuration.
"""

import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta

from models.api_token_scope import (
    ApiScope,
    ApiTokenConfig,
    ApiTokenScope,
)
from services.api_token_scope_service import (
    ApiTokenScopeService,
    ScopeNotAllowedError,
    TokenExpiredError,
    TokenIPRestrictionError,
)


class TestScopeManagement:
    """Tests for scope management operations."""

    @patch("services.api_token_scope_service.db")
    def test_get_token_scopes(self, mock_db):
        """Test getting all scopes for a token."""
        mock_db.session.scalars.return_value.all.return_value = [
            "app:invoke",
            "conversation:read",
        ]

        scopes = ApiTokenScopeService.get_token_scopes("token-123")

        assert len(scopes) == 2
        assert "app:invoke" in scopes
        assert "conversation:read" in scopes

    @patch("services.api_token_scope_service.db")
    def test_has_scope_true(self, mock_db):
        """Test checking if token has a specific scope."""
        # First call for admin check returns None
        # Second call for actual scope returns a result
        mock_db.session.scalar.side_effect = [None, "some-id"]

        result = ApiTokenScopeService.has_scope("token-123", ApiScope.APP_INVOKE)

        assert result is True

    @patch("services.api_token_scope_service.db")
    def test_has_scope_false(self, mock_db):
        """Test checking if token doesn't have a scope."""
        mock_db.session.scalar.side_effect = [None, None]

        result = ApiTokenScopeService.has_scope("token-123", ApiScope.ADMIN_ALL)

        assert result is False

    @patch("services.api_token_scope_service.db")
    def test_has_scope_admin_grants_all(self, mock_db):
        """Test that admin:all scope grants access to any scope."""
        # First call returns admin scope
        mock_db.session.scalar.return_value = "admin-scope-id"

        result = ApiTokenScopeService.has_scope("token-123", ApiScope.APP_INVOKE)

        assert result is True
        # Only one call needed since admin check passed
        assert mock_db.session.scalar.call_count == 1

    @patch("services.api_token_scope_service.db")
    def test_grant_scope(self, mock_db):
        """Test granting a scope to a token."""
        mock_db.session.scalar.return_value = None  # No existing scope
        mock_db.session.add = MagicMock()
        mock_db.session.commit = MagicMock()

        result = ApiTokenScopeService.grant_scope("token-123", ApiScope.APP_INVOKE)

        assert result.token_id == "token-123"
        assert result.scope == "app:invoke"
        mock_db.session.add.assert_called_once()

    @patch("services.api_token_scope_service.db")
    def test_grant_scope_already_exists(self, mock_db):
        """Test granting a scope that already exists returns existing."""
        existing_scope = MagicMock(spec=ApiTokenScope)
        existing_scope.token_id = "token-123"
        existing_scope.scope = "app:invoke"
        mock_db.session.scalar.return_value = existing_scope

        result = ApiTokenScopeService.grant_scope("token-123", ApiScope.APP_INVOKE)

        assert result == existing_scope
        mock_db.session.add.assert_not_called()

    @patch("services.api_token_scope_service.db")
    def test_revoke_scope(self, mock_db):
        """Test revoking a scope from a token."""
        mock_scope = MagicMock(spec=ApiTokenScope)
        mock_db.session.scalar.return_value = mock_scope
        mock_db.session.delete = MagicMock()
        mock_db.session.commit = MagicMock()

        result = ApiTokenScopeService.revoke_scope("token-123", ApiScope.APP_INVOKE)

        assert result is True
        mock_db.session.delete.assert_called_once_with(mock_scope)

    @patch("services.api_token_scope_service.db")
    def test_revoke_scope_not_found(self, mock_db):
        """Test revoking a scope that doesn't exist."""
        mock_db.session.scalar.return_value = None

        result = ApiTokenScopeService.revoke_scope("token-123", ApiScope.APP_INVOKE)

        assert result is False


class TestScopeValidation:
    """Tests for scope validation."""

    @patch("services.api_token_scope_service.db")
    def test_validate_token_scope_success(self, mock_db):
        """Test successful scope validation."""
        mock_db.session.scalar.side_effect = [None, "scope-id"]

        # Should not raise
        ApiTokenScopeService.validate_token_scope("token-123", ApiScope.APP_INVOKE)

    @patch("services.api_token_scope_service.db")
    def test_validate_token_scope_failure(self, mock_db):
        """Test scope validation failure."""
        mock_db.session.scalar.side_effect = [None, None]

        with pytest.raises(ScopeNotAllowedError) as exc_info:
            ApiTokenScopeService.validate_token_scope("token-123", ApiScope.APP_INVOKE)

        assert "app:invoke" in str(exc_info.value)

    @patch("services.api_token_scope_service.db")
    def test_has_any_scope_true(self, mock_db):
        """Test has_any_scope returns True when token has one of the scopes."""
        mock_db.session.scalar.side_effect = [None, "scope-id"]

        result = ApiTokenScopeService.has_any_scope(
            "token-123",
            [ApiScope.APP_INVOKE, ApiScope.WORKFLOW_RUN]
        )

        assert result is True

    @patch("services.api_token_scope_service.db")
    def test_has_all_scopes_true(self, mock_db):
        """Test has_all_scopes returns True when token has all scopes."""
        # Admin check fails, then two scope checks pass
        mock_db.session.scalar.side_effect = [None, "scope-1", None, "scope-2"]

        result = ApiTokenScopeService.has_all_scopes(
            "token-123",
            [ApiScope.APP_INVOKE, ApiScope.CONVERSATION_READ]
        )

        assert result is True

    @patch("services.api_token_scope_service.db")
    def test_has_all_scopes_false(self, mock_db):
        """Test has_all_scopes returns False when token is missing a scope."""
        # Admin check fails, first scope passes, second fails
        mock_db.session.scalar.side_effect = [None, "scope-1", None, None]

        result = ApiTokenScopeService.has_all_scopes(
            "token-123",
            [ApiScope.APP_INVOKE, ApiScope.ADMIN_ALL]
        )

        assert result is False


class TestTokenConfig:
    """Tests for token configuration."""

    @patch("services.api_token_scope_service.db")
    def test_create_config(self, mock_db):
        """Test creating token configuration."""
        mock_db.session.scalar.return_value = None  # No existing config
        mock_db.session.add = MagicMock()
        mock_db.session.commit = MagicMock()

        config = ApiTokenScopeService.create_or_update_config(
            token_id="token-123",
            name="My Token",
            description="Test token",
            rate_limit_rpm=100,
            allowed_ips=["192.168.1.0/24"],
        )

        assert config.token_id == "token-123"
        assert config.name == "My Token"
        assert config.rate_limit_rpm == 100

    @patch("services.api_token_scope_service.db")
    def test_update_config(self, mock_db):
        """Test updating existing token configuration."""
        existing_config = MagicMock(spec=ApiTokenConfig)
        existing_config.token_id = "token-123"
        mock_db.session.scalar.return_value = existing_config
        mock_db.session.commit = MagicMock()

        config = ApiTokenScopeService.create_or_update_config(
            token_id="token-123",
            name="Updated Name",
        )

        assert existing_config.name == "Updated Name"


class TestConfigValidation:
    """Tests for configuration validation."""

    @patch("services.api_token_scope_service.db")
    def test_validate_config_no_config(self, mock_db):
        """Test validation passes when no config exists."""
        mock_db.session.scalar.return_value = None

        result = ApiTokenScopeService.validate_token_config(
            "token-123",
            ip_address="192.168.1.1"
        )

        assert result is None

    @patch("services.api_token_scope_service.db")
    def test_validate_config_expired(self, mock_db):
        """Test validation fails for expired token."""
        mock_config = MagicMock(spec=ApiTokenConfig)
        mock_config.is_token_expired = True
        mock_db.session.scalar.return_value = mock_config

        with pytest.raises(TokenExpiredError):
            ApiTokenScopeService.validate_token_config("token-123")

    @patch("services.api_token_scope_service.db")
    def test_validate_config_ip_blocked(self, mock_db):
        """Test validation fails for blocked IP."""
        mock_config = MagicMock(spec=ApiTokenConfig)
        mock_config.is_token_expired = False
        mock_config.check_ip_allowed.return_value = False
        mock_config.allowed_referrers = None
        mock_db.session.scalar.return_value = mock_config

        with pytest.raises(TokenIPRestrictionError):
            ApiTokenScopeService.validate_token_config(
                "token-123",
                ip_address="10.0.0.1"
            )


class TestDefaultScopes:
    """Tests for default scope presets."""

    def test_get_default_scopes(self):
        """Test default scopes include expected permissions."""
        default_scopes = ApiScope.get_default_scopes()

        assert ApiScope.APP_INVOKE in default_scopes
        assert ApiScope.CONVERSATION_READ in default_scopes
        assert ApiScope.MESSAGE_READ in default_scopes
        assert ApiScope.ADMIN_ALL not in default_scopes

    def test_get_read_only_scopes(self):
        """Test read-only scopes don't include write permissions."""
        read_only = ApiScope.get_read_only_scopes()

        assert ApiScope.CONVERSATION_READ in read_only
        assert ApiScope.MESSAGE_READ in read_only
        assert ApiScope.CONVERSATION_WRITE not in read_only
        assert ApiScope.APP_INVOKE not in read_only

    def test_get_invoke_scopes(self):
        """Test invoke scopes include only invocation-related permissions."""
        invoke_scopes = ApiScope.get_invoke_scopes()

        assert ApiScope.APP_INVOKE in invoke_scopes
        assert ApiScope.APP_INVOKE_STREAM in invoke_scopes
        assert ApiScope.META_READ in invoke_scopes
        assert ApiScope.CONVERSATION_DELETE not in invoke_scopes


class TestScopeDescriptions:
    """Tests for scope information methods."""

    def test_get_all_scopes(self):
        """Test getting all available scopes."""
        all_scopes = ApiTokenScopeService.get_all_scopes()

        assert "app:invoke" in all_scopes
        assert "admin:all" in all_scopes
        assert len(all_scopes) == len(ApiScope)

    def test_get_scope_description(self):
        """Test getting scope descriptions."""
        desc = ApiTokenScopeService.get_scope_description(ApiScope.APP_INVOKE)

        assert "invoke" in desc.lower()
        assert len(desc) > 0

    def test_get_scope_description_unknown(self):
        """Test getting description for unknown scope."""
        desc = ApiTokenScopeService.get_scope_description("unknown:scope")

        assert "unknown" in desc.lower()


class TestUsageTracking:
    """Tests for usage tracking."""

    @patch("services.api_token_scope_service.db")
    def test_record_usage(self, mock_db):
        """Test recording token usage."""
        mock_config = MagicMock(spec=ApiTokenConfig)
        mock_config.total_requests = 10
        mock_config.total_tokens_used = 1000
        mock_db.session.scalar.return_value = mock_config
        mock_db.session.commit = MagicMock()

        ApiTokenScopeService.record_token_usage(
            "token-123",
            ip_address="192.168.1.1",
            tokens_used=500,
        )

        assert mock_config.total_requests == 11
        assert mock_config.total_tokens_used == 1500
        assert mock_config.last_used_ip == "192.168.1.1"

    @patch("services.api_token_scope_service.db")
    def test_record_usage_no_config(self, mock_db):
        """Test recording usage when no config exists is safe."""
        mock_db.session.scalar.return_value = None

        # Should not raise
        ApiTokenScopeService.record_token_usage("token-123")


class TestIPValidation:
    """Tests for IP address validation."""

    def test_ip_allowed_no_restrictions(self):
        """Test IP is allowed when no restrictions set."""
        config = ApiTokenConfig(
            token_id="token-123",
            allowed_ips=None,
        )

        assert config.check_ip_allowed("192.168.1.1") is True

    def test_ip_allowed_single_ip(self):
        """Test IP is allowed when it matches allowed IP."""
        import json
        config = ApiTokenConfig(
            token_id="token-123",
            allowed_ips=json.dumps(["192.168.1.1"]),
        )

        assert config.check_ip_allowed("192.168.1.1") is True
        assert config.check_ip_allowed("192.168.1.2") is False

    def test_ip_allowed_cidr(self):
        """Test IP is allowed within CIDR range."""
        import json
        config = ApiTokenConfig(
            token_id="token-123",
            allowed_ips=json.dumps(["192.168.1.0/24"]),
        )

        assert config.check_ip_allowed("192.168.1.100") is True
        assert config.check_ip_allowed("192.168.2.1") is False


class TestReferrerValidation:
    """Tests for referrer validation."""

    def test_referrer_allowed_no_restrictions(self):
        """Test referrer is allowed when no restrictions set."""
        config = ApiTokenConfig(
            token_id="token-123",
            allowed_referrers=None,
        )

        assert config.check_referrer_allowed("https://example.com") is True

    def test_referrer_allowed_pattern(self):
        """Test referrer is allowed when it matches pattern."""
        import json
        config = ApiTokenConfig(
            token_id="token-123",
            allowed_referrers=json.dumps(["*.example.com"]),
        )

        assert config.check_referrer_allowed("https://app.example.com") is True
        assert config.check_referrer_allowed("https://other.com") is False

    def test_referrer_required_but_missing(self):
        """Test referrer is blocked when required but not provided."""
        import json
        config = ApiTokenConfig(
            token_id="token-123",
            allowed_referrers=json.dumps(["*.example.com"]),
        )

        assert config.check_referrer_allowed(None) is False
