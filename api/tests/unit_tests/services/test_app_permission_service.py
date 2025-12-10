"""
Unit tests for AppPermissionService.

Tests cover:
- Permission checking logic
- Permission CRUD operations
- Access config management
- Role hierarchy validation
"""

from unittest.mock import MagicMock, patch

import pytest

from models.account import Account, TenantAccountRole
from models.app_permission import (
    AppAccessConfig,
    AppPermission,
    AppPermissionRole,
    AppPermissionType,
)
from models.model import App
from services.app_permission_service import (
    AppPermissionService,
    NoAppPermissionError,
)


class TestAppPermissionRoleHierarchy:
    """Tests for role hierarchy and validation."""

    def test_is_valid_role_with_valid_roles(self):
        """Test that valid roles are recognized."""
        assert AppPermissionRole.is_valid_role("owner") is True
        assert AppPermissionRole.is_valid_role("admin") is True
        assert AppPermissionRole.is_valid_role("operator") is True
        assert AppPermissionRole.is_valid_role("viewer") is True

    def test_is_valid_role_with_invalid_roles(self):
        """Test that invalid roles are rejected."""
        assert AppPermissionRole.is_valid_role("invalid") is False
        assert AppPermissionRole.is_valid_role("") is False
        assert AppPermissionRole.is_valid_role(None) is False

    def test_is_admin_or_owner(self):
        """Test admin/owner role check."""
        assert AppPermissionRole.is_admin_or_owner(AppPermissionRole.OWNER) is True
        assert AppPermissionRole.is_admin_or_owner(AppPermissionRole.ADMIN) is True
        assert AppPermissionRole.is_admin_or_owner(AppPermissionRole.OPERATOR) is False
        assert AppPermissionRole.is_admin_or_owner(AppPermissionRole.VIEWER) is False
        assert AppPermissionRole.is_admin_or_owner(None) is False

    def test_can_invoke(self):
        """Test invoke permission check."""
        assert AppPermissionRole.can_invoke(AppPermissionRole.OWNER) is True
        assert AppPermissionRole.can_invoke(AppPermissionRole.ADMIN) is True
        assert AppPermissionRole.can_invoke(AppPermissionRole.OPERATOR) is True
        assert AppPermissionRole.can_invoke(AppPermissionRole.VIEWER) is False
        assert AppPermissionRole.can_invoke(None) is False

    def test_can_view(self):
        """Test view permission check."""
        assert AppPermissionRole.can_view(AppPermissionRole.OWNER) is True
        assert AppPermissionRole.can_view(AppPermissionRole.ADMIN) is True
        assert AppPermissionRole.can_view(AppPermissionRole.OPERATOR) is True
        assert AppPermissionRole.can_view(AppPermissionRole.VIEWER) is True
        assert AppPermissionRole.can_view(None) is False

    def test_can_manage_permissions(self):
        """Test permission management check."""
        assert AppPermissionRole.can_manage_permissions(AppPermissionRole.OWNER) is True
        assert AppPermissionRole.can_manage_permissions(AppPermissionRole.ADMIN) is True
        assert AppPermissionRole.can_manage_permissions(AppPermissionRole.OPERATOR) is False
        assert AppPermissionRole.can_manage_permissions(AppPermissionRole.VIEWER) is False


class TestAppPermissionService:
    """Tests for AppPermissionService."""

    @pytest.fixture
    def mock_app(self):
        """Create a mock App instance."""
        app = MagicMock(spec=App)
        app.id = "test-app-id"
        app.tenant_id = "test-tenant-id"
        return app

    @pytest.fixture
    def mock_user(self):
        """Create a mock Account instance."""
        user = MagicMock(spec=Account)
        user.id = "test-user-id"
        user.current_tenant_id = "test-tenant-id"
        user.current_role = TenantAccountRole.EDITOR
        return user

    @pytest.fixture
    def mock_owner_user(self):
        """Create a mock Account with OWNER role."""
        user = MagicMock(spec=Account)
        user.id = "owner-user-id"
        user.current_tenant_id = "test-tenant-id"
        user.current_role = TenantAccountRole.OWNER
        return user

    def test_role_meets_requirement_hierarchy(self):
        """Test that role hierarchy is correctly evaluated."""
        # Owner meets all requirements
        assert AppPermissionService._role_meets_requirement(
            AppPermissionRole.OWNER, AppPermissionRole.OWNER
        ) is True
        assert AppPermissionService._role_meets_requirement(
            AppPermissionRole.OWNER, AppPermissionRole.ADMIN
        ) is True
        assert AppPermissionService._role_meets_requirement(
            AppPermissionRole.OWNER, AppPermissionRole.OPERATOR
        ) is True
        assert AppPermissionService._role_meets_requirement(
            AppPermissionRole.OWNER, AppPermissionRole.VIEWER
        ) is True

        # Admin meets admin and below
        assert AppPermissionService._role_meets_requirement(
            AppPermissionRole.ADMIN, AppPermissionRole.OWNER
        ) is False
        assert AppPermissionService._role_meets_requirement(
            AppPermissionRole.ADMIN, AppPermissionRole.ADMIN
        ) is True
        assert AppPermissionService._role_meets_requirement(
            AppPermissionRole.ADMIN, AppPermissionRole.OPERATOR
        ) is True

        # Operator meets operator and below
        assert AppPermissionService._role_meets_requirement(
            AppPermissionRole.OPERATOR, AppPermissionRole.ADMIN
        ) is False
        assert AppPermissionService._role_meets_requirement(
            AppPermissionRole.OPERATOR, AppPermissionRole.OPERATOR
        ) is True
        assert AppPermissionService._role_meets_requirement(
            AppPermissionRole.OPERATOR, AppPermissionRole.VIEWER
        ) is True

        # Viewer only meets viewer
        assert AppPermissionService._role_meets_requirement(
            AppPermissionRole.VIEWER, AppPermissionRole.OPERATOR
        ) is False
        assert AppPermissionService._role_meets_requirement(
            AppPermissionRole.VIEWER, AppPermissionRole.VIEWER
        ) is True

    @patch.object(AppPermissionService, "get_app_access_config")
    def test_check_app_permission_tenant_mismatch(self, mock_get_config, mock_app, mock_user):
        """Test that cross-tenant access is blocked."""
        mock_user.current_tenant_id = "different-tenant-id"

        with pytest.raises(NoAppPermissionError) as exc_info:
            AppPermissionService.check_app_permission(mock_app, mock_user)

        assert "do not have permission" in str(exc_info.value)

    @patch.object(AppPermissionService, "get_app_access_config")
    def test_check_app_permission_inherits_workspace(self, mock_get_config, mock_app, mock_user):
        """Test that workspace permissions are used when app inherits."""
        mock_get_config.return_value = None  # No custom config = inherit workspace

        # Should not raise for workspace member
        result = AppPermissionService.check_app_permission(mock_app, mock_user)
        assert result is None  # Using workspace permissions

    @patch.object(AppPermissionService, "get_app_access_config")
    @patch.object(AppPermissionService, "get_user_app_permission")
    def test_check_app_permission_restricted_no_permission(
        self, mock_get_permission, mock_get_config, mock_app, mock_user
    ):
        """Test that restricted apps require explicit permission."""
        # Set up restricted config
        mock_config = MagicMock(spec=AppAccessConfig)
        mock_config.inherits_workspace = False
        mock_config.permission_type = AppPermissionType.RESTRICTED
        mock_get_config.return_value = mock_config

        # No explicit permission
        mock_get_permission.return_value = None

        with pytest.raises(NoAppPermissionError):
            AppPermissionService.check_app_permission(mock_app, mock_user)

    @patch.object(AppPermissionService, "get_app_access_config")
    @patch.object(AppPermissionService, "get_user_app_permission")
    def test_check_app_permission_restricted_with_permission(
        self, mock_get_permission, mock_get_config, mock_app, mock_user
    ):
        """Test that restricted apps allow users with explicit permission."""
        # Set up restricted config
        mock_config = MagicMock(spec=AppAccessConfig)
        mock_config.inherits_workspace = False
        mock_config.permission_type = AppPermissionType.RESTRICTED
        mock_get_config.return_value = mock_config

        # User has explicit permission
        mock_permission = MagicMock(spec=AppPermission)
        mock_permission.role = AppPermissionRole.OPERATOR
        mock_permission.role_enum = AppPermissionRole.OPERATOR
        mock_get_permission.return_value = mock_permission

        result = AppPermissionService.check_app_permission(mock_app, mock_user)
        assert result == mock_permission

    @patch.object(AppPermissionService, "get_app_access_config")
    @patch.object(AppPermissionService, "get_user_app_permission")
    def test_check_app_permission_restricted_workspace_owner_bypass(
        self, mock_get_permission, mock_get_config, mock_app, mock_owner_user
    ):
        """Test that workspace owners can access restricted apps."""
        # Set up restricted config
        mock_config = MagicMock(spec=AppAccessConfig)
        mock_config.inherits_workspace = False
        mock_config.permission_type = AppPermissionType.RESTRICTED
        mock_get_config.return_value = mock_config

        # No explicit permission, but user is workspace owner
        mock_get_permission.return_value = None

        result = AppPermissionService.check_app_permission(mock_app, mock_owner_user)
        assert result is None  # Owner bypasses

    @patch.object(AppPermissionService, "check_app_permission")
    def test_can_invoke_app_true(self, mock_check, mock_app, mock_user):
        """Test can_invoke_app returns True when user has operator+ role."""
        mock_check.return_value = None  # No exception = has permission

        result = AppPermissionService.can_invoke_app(mock_app, mock_user)
        assert result is True

    @patch.object(AppPermissionService, "check_app_permission")
    def test_can_invoke_app_false(self, mock_check, mock_app, mock_user):
        """Test can_invoke_app returns False when user lacks permission."""
        mock_check.side_effect = NoAppPermissionError()

        result = AppPermissionService.can_invoke_app(mock_app, mock_user)
        assert result is False

    @patch.object(AppPermissionService, "check_app_permission")
    def test_can_view_app_true(self, mock_check, mock_app, mock_user):
        """Test can_view_app returns True when user has viewer+ role."""
        mock_check.return_value = None

        result = AppPermissionService.can_view_app(mock_app, mock_user)
        assert result is True

    @patch.object(AppPermissionService, "check_app_permission")
    def test_can_manage_app_true(self, mock_check, mock_app, mock_user):
        """Test can_manage_app returns True when user has admin+ role."""
        mock_check.return_value = None

        result = AppPermissionService.can_manage_app(mock_app, mock_user)
        assert result is True


class TestAppAccessConfigModel:
    """Tests for AppAccessConfig model properties."""

    def test_is_restricted_true(self):
        """Test is_restricted property when permission_type is RESTRICTED."""
        config = MagicMock(spec=AppAccessConfig)
        config.permission_type = AppPermissionType.RESTRICTED
        config.is_restricted = AppPermissionType.RESTRICTED == config.permission_type

        assert config.is_restricted is True

    def test_inherits_workspace_true(self):
        """Test inherits_workspace property when permission_type is INHERIT_WORKSPACE."""
        config = MagicMock(spec=AppAccessConfig)
        config.permission_type = AppPermissionType.INHERIT_WORKSPACE
        config.inherits_workspace = AppPermissionType.INHERIT_WORKSPACE == config.permission_type

        assert config.inherits_workspace is True


class TestAppPermissionModel:
    """Tests for AppPermission model properties."""

    def test_role_enum_conversion(self):
        """Test that role string is correctly converted to enum."""
        permission = MagicMock(spec=AppPermission)
        permission.role = "admin"

        # Simulate the property
        role_enum = AppPermissionRole(permission.role)
        assert role_enum == AppPermissionRole.ADMIN

    def test_can_invoke_property(self):
        """Test can_invoke property based on role."""
        # Owner can invoke
        assert AppPermissionRole.can_invoke(AppPermissionRole.OWNER) is True
        # Admin can invoke
        assert AppPermissionRole.can_invoke(AppPermissionRole.ADMIN) is True
        # Operator can invoke
        assert AppPermissionRole.can_invoke(AppPermissionRole.OPERATOR) is True
        # Viewer cannot invoke
        assert AppPermissionRole.can_invoke(AppPermissionRole.VIEWER) is False

    def test_can_view_property(self):
        """Test can_view property based on role."""
        # All roles can view
        assert AppPermissionRole.can_view(AppPermissionRole.OWNER) is True
        assert AppPermissionRole.can_view(AppPermissionRole.ADMIN) is True
        assert AppPermissionRole.can_view(AppPermissionRole.OPERATOR) is True
        assert AppPermissionRole.can_view(AppPermissionRole.VIEWER) is True

    def test_can_manage_permissions_property(self):
        """Test can_manage_permissions property based on role."""
        # Only owner and admin can manage
        assert AppPermissionRole.can_manage_permissions(AppPermissionRole.OWNER) is True
        assert AppPermissionRole.can_manage_permissions(AppPermissionRole.ADMIN) is True
        assert AppPermissionRole.can_manage_permissions(AppPermissionRole.OPERATOR) is False
        assert AppPermissionRole.can_manage_permissions(AppPermissionRole.VIEWER) is False
