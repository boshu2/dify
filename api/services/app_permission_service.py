"""
App Permission Service - Business logic for per-app RBAC.

This service provides methods for managing app-level permissions,
checking access rights, and handling permission grants/revocations.
"""

from collections.abc import Sequence

from sqlalchemy import select

from extensions.ext_database import db
from models.account import Account, TenantAccountRole
from models.app_permission import (
    AppAccessConfig,
    AppPermission,
    AppPermissionRole,
    AppPermissionType,
)
from models.model import App


class NoAppPermissionError(Exception):
    """Raised when a user does not have permission to access an app."""

    def __init__(self, message: str = "You do not have permission to access this app."):
        self.message = message
        super().__init__(self.message)


class AppPermissionService:
    """Service for managing app-level permissions."""

    # ==========================================
    # Permission Checking Methods
    # ==========================================

    @classmethod
    def check_app_permission(
        cls,
        app: App,
        user: Account,
        required_role: AppPermissionRole | None = None,
    ) -> AppPermission | None:
        """
        Check if a user has permission to access an app.

        Args:
            app: The app to check access for
            user: The user requesting access
            required_role: Minimum role required (if None, any access is sufficient)

        Returns:
            AppPermission if explicit permission exists, None if using workspace permissions

        Raises:
            NoAppPermissionError: If user does not have sufficient permissions
        """
        # First check tenant boundary
        if app.tenant_id != user.current_tenant_id:
            raise NoAppPermissionError("You do not have permission to access this app.")

        # Get access config for the app
        access_config = cls.get_app_access_config(app.id)

        # If app inherits workspace permissions, check workspace role
        if not access_config or access_config.inherits_workspace:
            return cls._check_workspace_permission(app, user, required_role)

        # For restricted apps, check explicit permission
        permission = cls.get_user_app_permission(app.id, user.id)

        if not permission:
            # Workspace owners always have access
            if user.current_role == TenantAccountRole.OWNER:
                return None
            raise NoAppPermissionError("You do not have permission to access this app.")

        # Check if the role meets the requirement
        if required_role:
            if not cls._role_meets_requirement(permission.role_enum, required_role):
                raise NoAppPermissionError(
                    f"Your role ({permission.role}) does not have sufficient permissions."
                )

        return permission

    @classmethod
    def _check_workspace_permission(
        cls,
        app: App,
        user: Account,
        required_role: AppPermissionRole | None,
    ) -> None:
        """Check workspace-level permissions (backward compatible)."""
        # Map workspace roles to app permission roles
        workspace_to_app_role = {
            TenantAccountRole.OWNER: AppPermissionRole.OWNER,
            TenantAccountRole.ADMIN: AppPermissionRole.ADMIN,
            TenantAccountRole.EDITOR: AppPermissionRole.OPERATOR,
            TenantAccountRole.NORMAL: AppPermissionRole.VIEWER,
            TenantAccountRole.DATASET_OPERATOR: AppPermissionRole.VIEWER,
        }

        effective_role = workspace_to_app_role.get(user.current_role)
        if not effective_role:
            raise NoAppPermissionError("Invalid workspace role.")

        if required_role and not cls._role_meets_requirement(effective_role, required_role):
            raise NoAppPermissionError(
                f"Your workspace role does not have sufficient permissions for this operation."
            )

        return None

    @staticmethod
    def _role_meets_requirement(
        user_role: AppPermissionRole,
        required_role: AppPermissionRole,
    ) -> bool:
        """Check if a user role meets the required permission level."""
        role_hierarchy = {
            AppPermissionRole.OWNER: 4,
            AppPermissionRole.ADMIN: 3,
            AppPermissionRole.OPERATOR: 2,
            AppPermissionRole.VIEWER: 1,
        }
        return role_hierarchy.get(user_role, 0) >= role_hierarchy.get(required_role, 0)

    @classmethod
    def can_invoke_app(cls, app: App, user: Account) -> bool:
        """Check if user can invoke/run the app."""
        try:
            cls.check_app_permission(app, user, AppPermissionRole.OPERATOR)
            return True
        except NoAppPermissionError:
            return False

    @classmethod
    def can_view_app(cls, app: App, user: Account) -> bool:
        """Check if user can view app details."""
        try:
            cls.check_app_permission(app, user, AppPermissionRole.VIEWER)
            return True
        except NoAppPermissionError:
            return False

    @classmethod
    def can_manage_app(cls, app: App, user: Account) -> bool:
        """Check if user can manage app settings."""
        try:
            cls.check_app_permission(app, user, AppPermissionRole.ADMIN)
            return True
        except NoAppPermissionError:
            return False

    # ==========================================
    # Permission CRUD Methods
    # ==========================================

    @classmethod
    def get_user_app_permission(cls, app_id: str, account_id: str) -> AppPermission | None:
        """Get a specific user's permission for an app."""
        return db.session.scalar(
            select(AppPermission).where(
                AppPermission.app_id == app_id,
                AppPermission.account_id == account_id,
            )
        )

    @classmethod
    def get_app_permissions(cls, app_id: str) -> Sequence[AppPermission]:
        """Get all permission grants for an app."""
        return db.session.scalars(
            select(AppPermission).where(AppPermission.app_id == app_id)
        ).all()

    @classmethod
    def get_user_accessible_apps(
        cls,
        tenant_id: str,
        account_id: str,
        include_workspace_inherited: bool = True,
    ) -> Sequence[str]:
        """
        Get list of app IDs the user can access.

        Args:
            tenant_id: The workspace ID
            account_id: The user's account ID
            include_workspace_inherited: Whether to include apps that inherit workspace permissions

        Returns:
            List of app IDs the user can access
        """
        # Get explicitly granted app IDs
        explicit_app_ids = db.session.scalars(
            select(AppPermission.app_id).where(
                AppPermission.tenant_id == tenant_id,
                AppPermission.account_id == account_id,
            )
        ).all()

        if not include_workspace_inherited:
            return explicit_app_ids

        # Get apps that inherit workspace permissions
        inherited_app_ids = db.session.scalars(
            select(App.id).where(
                App.tenant_id == tenant_id,
                App.status == "normal",
            ).where(
                ~App.id.in_(
                    select(AppAccessConfig.app_id).where(
                        AppAccessConfig.permission_type == AppPermissionType.RESTRICTED
                    )
                )
            )
        ).all()

        return list(set(explicit_app_ids) | set(inherited_app_ids))

    @classmethod
    def grant_permission(
        cls,
        app: App,
        account_id: str,
        role: AppPermissionRole,
        granted_by: str,
    ) -> AppPermission:
        """
        Grant or update permission for a user on an app.

        Args:
            app: The app to grant permission for
            account_id: The user to grant permission to
            role: The role to grant
            granted_by: ID of the user granting the permission

        Returns:
            The created or updated AppPermission
        """
        existing = cls.get_user_app_permission(app.id, account_id)

        if existing:
            existing.role = role.value
            existing.granted_by = granted_by
            db.session.commit()
            return existing

        permission = AppPermission(
            app_id=app.id,
            account_id=account_id,
            tenant_id=app.tenant_id,
            role=role.value,
            granted_by=granted_by,
        )
        db.session.add(permission)
        db.session.commit()
        return permission

    @classmethod
    def revoke_permission(cls, app_id: str, account_id: str) -> bool:
        """
        Revoke a user's permission for an app.

        Args:
            app_id: The app ID
            account_id: The user ID to revoke permission from

        Returns:
            True if permission was revoked, False if no permission existed
        """
        permission = cls.get_user_app_permission(app_id, account_id)
        if not permission:
            return False

        db.session.delete(permission)
        db.session.commit()
        return True

    @classmethod
    def bulk_grant_permissions(
        cls,
        app: App,
        account_ids: list[str],
        role: AppPermissionRole,
        granted_by: str,
    ) -> list[AppPermission]:
        """
        Grant permissions to multiple users at once.

        Args:
            app: The app to grant permissions for
            account_ids: List of user IDs to grant permission to
            role: The role to grant
            granted_by: ID of the user granting the permissions

        Returns:
            List of created/updated AppPermissions
        """
        permissions = []
        for account_id in account_ids:
            permission = cls.grant_permission(app, account_id, role, granted_by)
            permissions.append(permission)
        return permissions

    @classmethod
    def clear_app_permissions(cls, app_id: str) -> int:
        """
        Remove all permission grants for an app.

        Args:
            app_id: The app ID

        Returns:
            Number of permissions removed
        """
        result = db.session.execute(
            AppPermission.__table__.delete().where(AppPermission.app_id == app_id)
        )
        db.session.commit()
        return result.rowcount

    # ==========================================
    # Access Config Methods
    # ==========================================

    @classmethod
    def get_app_access_config(cls, app_id: str) -> AppAccessConfig | None:
        """Get the access configuration for an app."""
        return db.session.scalar(
            select(AppAccessConfig).where(AppAccessConfig.app_id == app_id)
        )

    @classmethod
    def create_or_update_access_config(
        cls,
        app: App,
        permission_type: AppPermissionType,
        require_api_scope: bool = False,
        custom_rate_limit_rpm: int | None = None,
        custom_rate_limit_rph: int | None = None,
        access_description: str | None = None,
    ) -> AppAccessConfig:
        """
        Create or update the access configuration for an app.

        Args:
            app: The app to configure
            permission_type: The permission type (INHERIT_WORKSPACE, RESTRICTED, PUBLIC_READ)
            require_api_scope: Whether to require specific API scopes
            custom_rate_limit_rpm: Custom requests per minute limit
            custom_rate_limit_rph: Custom requests per hour limit
            access_description: Description of access requirements

        Returns:
            The created or updated AppAccessConfig
        """
        existing = cls.get_app_access_config(app.id)

        if existing:
            existing.permission_type = permission_type.value
            existing.require_api_scope = require_api_scope
            existing.custom_rate_limit_rpm = custom_rate_limit_rpm
            existing.custom_rate_limit_rph = custom_rate_limit_rph
            existing.access_description = access_description
            db.session.commit()
            return existing

        config = AppAccessConfig(
            app_id=app.id,
            tenant_id=app.tenant_id,
            permission_type=permission_type.value,
            require_api_scope=require_api_scope,
            custom_rate_limit_rpm=custom_rate_limit_rpm,
            custom_rate_limit_rph=custom_rate_limit_rph,
            access_description=access_description,
        )
        db.session.add(config)
        db.session.commit()
        return config

    @classmethod
    def set_app_restricted(cls, app: App) -> AppAccessConfig:
        """Set an app to restricted mode (requires explicit permissions)."""
        return cls.create_or_update_access_config(app, AppPermissionType.RESTRICTED)

    @classmethod
    def set_app_inherit_workspace(cls, app: App) -> AppAccessConfig:
        """Set an app to inherit workspace permissions (default behavior)."""
        return cls.create_or_update_access_config(app, AppPermissionType.INHERIT_WORKSPACE)

    @classmethod
    def delete_access_config(cls, app_id: str) -> bool:
        """
        Delete access configuration for an app (reverts to default behavior).

        Args:
            app_id: The app ID

        Returns:
            True if config was deleted, False if no config existed
        """
        config = cls.get_app_access_config(app_id)
        if not config:
            return False

        db.session.delete(config)
        db.session.commit()
        return True

    # ==========================================
    # Permission Transfer Methods
    # ==========================================

    @classmethod
    def transfer_ownership(
        cls,
        app: App,
        new_owner_id: str,
        current_owner_id: str,
    ) -> tuple[AppPermission, AppPermission | None]:
        """
        Transfer app ownership to another user.

        Args:
            app: The app to transfer
            new_owner_id: The new owner's account ID
            current_owner_id: The current owner's account ID

        Returns:
            Tuple of (new owner permission, old owner permission or None)
        """
        # Grant OWNER to new owner
        new_owner_permission = cls.grant_permission(
            app, new_owner_id, AppPermissionRole.OWNER, current_owner_id
        )

        # Demote current owner to ADMIN
        old_owner_permission = None
        if current_owner_id != new_owner_id:
            old_owner_permission = cls.grant_permission(
                app, current_owner_id, AppPermissionRole.ADMIN, current_owner_id
            )

        return new_owner_permission, old_owner_permission
