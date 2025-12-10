"""
App Permission Model - Per-app RBAC for OpenShift-for-AI model.

This module provides fine-grained access control at the app level,
independent of workspace-level permissions.
"""

import enum
from datetime import datetime
from uuid import uuid4

import sqlalchemy as sa
from sqlalchemy import DateTime, String, func
from sqlalchemy.orm import Mapped, mapped_column

from .base import TypeBase
from .types import LongText, StringUUID


class AppPermissionRole(enum.StrEnum):
    """
    Role levels for app-specific permissions.

    OWNER: Full control over the app (can delete, transfer ownership)
    ADMIN: Can manage app settings and permissions
    OPERATOR: Can invoke the app and view executions
    VIEWER: Read-only access to app details and logs
    """
    OWNER = "owner"
    ADMIN = "admin"
    OPERATOR = "operator"
    VIEWER = "viewer"

    @staticmethod
    def is_valid_role(role: str) -> bool:
        """Check if role string is a valid AppPermissionRole."""
        if not role:
            return False
        return role in {
            AppPermissionRole.OWNER,
            AppPermissionRole.ADMIN,
            AppPermissionRole.OPERATOR,
            AppPermissionRole.VIEWER,
        }

    @staticmethod
    def is_admin_or_owner(role: "AppPermissionRole | None") -> bool:
        """Check if role has admin-level permissions."""
        if not role:
            return False
        return role in {AppPermissionRole.OWNER, AppPermissionRole.ADMIN}

    @staticmethod
    def can_invoke(role: "AppPermissionRole | None") -> bool:
        """Check if role can invoke the app."""
        if not role:
            return False
        return role in {
            AppPermissionRole.OWNER,
            AppPermissionRole.ADMIN,
            AppPermissionRole.OPERATOR,
        }

    @staticmethod
    def can_view(role: "AppPermissionRole | None") -> bool:
        """Check if role can view app details."""
        if not role:
            return False
        return role in {
            AppPermissionRole.OWNER,
            AppPermissionRole.ADMIN,
            AppPermissionRole.OPERATOR,
            AppPermissionRole.VIEWER,
        }

    @staticmethod
    def can_manage_permissions(role: "AppPermissionRole | None") -> bool:
        """Check if role can manage other users' permissions."""
        if not role:
            return False
        return role in {AppPermissionRole.OWNER, AppPermissionRole.ADMIN}


class AppPermissionType(enum.StrEnum):
    """
    Permission type for app access control.

    INHERIT_WORKSPACE: Use workspace-level permissions (default for backward compatibility)
    RESTRICTED: Only users with explicit AppPermission can access
    PUBLIC_READ: Anyone in workspace can view, but invoke requires permission
    """
    INHERIT_WORKSPACE = "inherit_workspace"
    RESTRICTED = "restricted"
    PUBLIC_READ = "public_read"


class AppPermission(TypeBase):
    """
    Per-app permission grants for individual users.

    This table stores explicit permission grants for users to access
    specific apps, independent of their workspace role.
    """
    __tablename__ = "app_permissions"
    __table_args__ = (
        sa.PrimaryKeyConstraint("id", name="app_permission_pkey"),
        sa.Index("idx_app_permissions_app_id", "app_id"),
        sa.Index("idx_app_permissions_account_id", "account_id"),
        sa.Index("idx_app_permissions_tenant_id", "tenant_id"),
        sa.UniqueConstraint("app_id", "account_id", name="unique_app_permission_app_account"),
    )

    id: Mapped[str] = mapped_column(
        StringUUID,
        insert_default=lambda: str(uuid4()),
        default_factory=lambda: str(uuid4()),
        primary_key=True,
        init=False,
    )
    app_id: Mapped[str] = mapped_column(StringUUID, nullable=False)
    account_id: Mapped[str] = mapped_column(StringUUID, nullable=False)
    tenant_id: Mapped[str] = mapped_column(StringUUID, nullable=False)
    role: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        server_default=sa.text("'viewer'"),
        default=AppPermissionRole.VIEWER,
    )
    granted_by: Mapped[str | None] = mapped_column(StringUUID, nullable=True, default=None)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.current_timestamp(), init=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        server_default=func.current_timestamp(),
        onupdate=func.current_timestamp(),
        init=False,
    )

    def __repr__(self) -> str:
        return f"<AppPermission(app_id={self.app_id}, account_id={self.account_id}, role={self.role})>"

    @property
    def role_enum(self) -> AppPermissionRole:
        """Get role as enum."""
        return AppPermissionRole(self.role)

    @property
    def can_invoke(self) -> bool:
        """Check if this permission allows app invocation."""
        return AppPermissionRole.can_invoke(self.role_enum)

    @property
    def can_view(self) -> bool:
        """Check if this permission allows viewing app details."""
        return AppPermissionRole.can_view(self.role_enum)

    @property
    def can_manage_permissions(self) -> bool:
        """Check if this permission allows managing other users' permissions."""
        return AppPermissionRole.can_manage_permissions(self.role_enum)


class AppAccessConfig(TypeBase):
    """
    Per-app access configuration settings.

    This table stores the access control mode for each app,
    determining how permissions are evaluated.
    """
    __tablename__ = "app_access_configs"
    __table_args__ = (
        sa.PrimaryKeyConstraint("id", name="app_access_config_pkey"),
        sa.Index("idx_app_access_configs_app_id", "app_id"),
        sa.UniqueConstraint("app_id", name="unique_app_access_config_app_id"),
    )

    id: Mapped[str] = mapped_column(
        StringUUID,
        insert_default=lambda: str(uuid4()),
        default_factory=lambda: str(uuid4()),
        primary_key=True,
        init=False,
    )
    app_id: Mapped[str] = mapped_column(StringUUID, nullable=False)
    tenant_id: Mapped[str] = mapped_column(StringUUID, nullable=False)
    permission_type: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        server_default=sa.text("'inherit_workspace'"),
        default=AppPermissionType.INHERIT_WORKSPACE,
    )
    # Optional: require specific API scopes even for workspace members
    require_api_scope: Mapped[bool] = mapped_column(
        sa.Boolean,
        nullable=False,
        server_default=sa.text("false"),
        default=False,
    )
    # Optional: custom rate limits for this app (overrides workspace defaults)
    custom_rate_limit_rpm: Mapped[int | None] = mapped_column(sa.Integer, nullable=True, default=None)
    custom_rate_limit_rph: Mapped[int | None] = mapped_column(sa.Integer, nullable=True, default=None)
    # Optional: description of access requirements
    access_description: Mapped[str | None] = mapped_column(LongText, nullable=True, default=None)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.current_timestamp(), init=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        server_default=func.current_timestamp(),
        onupdate=func.current_timestamp(),
        init=False,
    )

    def __repr__(self) -> str:
        return f"<AppAccessConfig(app_id={self.app_id}, permission_type={self.permission_type})>"

    @property
    def permission_type_enum(self) -> AppPermissionType:
        """Get permission type as enum."""
        return AppPermissionType(self.permission_type)

    @property
    def is_restricted(self) -> bool:
        """Check if app requires explicit permissions."""
        return self.permission_type == AppPermissionType.RESTRICTED

    @property
    def inherits_workspace(self) -> bool:
        """Check if app uses workspace-level permissions."""
        return self.permission_type == AppPermissionType.INHERIT_WORKSPACE
