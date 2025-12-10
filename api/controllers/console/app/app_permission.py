"""
App Permission API Endpoints - Console API for per-app RBAC.

This module provides REST API endpoints for managing app-level permissions,
including granting, revoking, and listing permissions.
"""

from flask_restx import Resource, fields, marshal_with
from pydantic import BaseModel, Field
from werkzeug.exceptions import Forbidden, NotFound

from controllers.console import console_ns
from controllers.console.app.wraps import get_app_model
from controllers.console.wraps import (
    account_initialization_required,
    edit_permission_required,
    is_admin_or_owner_required,
    setup_required,
)
from extensions.ext_database import db
from libs.helper import TimestampField
from libs.login import current_account_with_tenant, login_required
from models.account import Account, TenantAccountRole
from models.app_permission import (
    AppAccessConfig,
    AppPermission,
    AppPermissionRole,
    AppPermissionType,
)
from services.app_permission_service import AppPermissionService, NoAppPermissionError


# ============================================
# Pydantic Request Models
# ============================================

class GrantPermissionPayload(BaseModel):
    """Payload for granting app permission to a user."""
    account_id: str = Field(..., description="Account ID to grant permission to")
    role: str = Field(
        default="viewer",
        description="Permission role: owner, admin, operator, viewer"
    )


class BulkGrantPermissionPayload(BaseModel):
    """Payload for granting permission to multiple users."""
    account_ids: list[str] = Field(..., description="List of account IDs")
    role: str = Field(
        default="viewer",
        description="Permission role: owner, admin, operator, viewer"
    )


class UpdateAccessConfigPayload(BaseModel):
    """Payload for updating app access configuration."""
    permission_type: str = Field(
        ...,
        description="Permission type: inherit_workspace, restricted, public_read"
    )
    require_api_scope: bool = Field(
        default=False,
        description="Whether to require specific API scopes"
    )
    custom_rate_limit_rpm: int | None = Field(
        default=None,
        description="Custom requests per minute limit"
    )
    custom_rate_limit_rph: int | None = Field(
        default=None,
        description="Custom requests per hour limit"
    )
    access_description: str | None = Field(
        default=None,
        description="Description of access requirements"
    )


# ============================================
# Flask-RESTX Response Fields
# ============================================

app_permission_fields = {
    "id": fields.String,
    "app_id": fields.String,
    "account_id": fields.String,
    "tenant_id": fields.String,
    "role": fields.String,
    "granted_by": fields.String,
    "created_at": TimestampField,
    "updated_at": TimestampField,
    # Nested account info
    "account": fields.Nested({
        "id": fields.String,
        "name": fields.String,
        "email": fields.String,
        "avatar": fields.String,
    }, allow_null=True),
}

app_permission_list_fields = {
    "permissions": fields.List(fields.Nested(app_permission_fields)),
    "total": fields.Integer,
}

app_access_config_fields = {
    "id": fields.String,
    "app_id": fields.String,
    "tenant_id": fields.String,
    "permission_type": fields.String,
    "require_api_scope": fields.Boolean,
    "custom_rate_limit_rpm": fields.Integer,
    "custom_rate_limit_rph": fields.Integer,
    "access_description": fields.String,
    "created_at": TimestampField,
    "updated_at": TimestampField,
}


# ============================================
# API Endpoints
# ============================================

@console_ns.route("/apps/<uuid:app_id>/permissions")
class AppPermissionListApi(Resource):
    """List and grant app permissions."""

    @setup_required
    @login_required
    @account_initialization_required
    @get_app_model
    @marshal_with(app_permission_list_fields)
    def get(self, app_model):
        """Get list of all permissions for an app."""
        current_user, _ = current_account_with_tenant()

        # Check if user can manage this app
        if not AppPermissionService.can_manage_app(app_model, current_user):
            raise Forbidden("You do not have permission to view app permissions.")

        permissions = AppPermissionService.get_app_permissions(app_model.id)

        # Enrich with account info
        result = []
        for perm in permissions:
            account = db.session.get(Account, perm.account_id)
            result.append({
                **perm.__dict__,
                "account": {
                    "id": account.id,
                    "name": account.name,
                    "email": account.email,
                    "avatar": account.avatar,
                } if account else None,
            })

        return {
            "permissions": result,
            "total": len(result),
        }

    @setup_required
    @login_required
    @account_initialization_required
    @get_app_model
    @marshal_with(app_permission_fields)
    def post(self, app_model):
        """Grant permission to a user for an app."""
        current_user, _ = current_account_with_tenant()

        # Check if user can manage this app
        if not AppPermissionService.can_manage_app(app_model, current_user):
            raise Forbidden("You do not have permission to manage app permissions.")

        payload = GrantPermissionPayload.model_validate(console_ns.payload)

        # Validate role
        if not AppPermissionRole.is_valid_role(payload.role):
            raise ValueError(f"Invalid role: {payload.role}")

        # Only owners can grant owner role
        if payload.role == AppPermissionRole.OWNER:
            user_permission = AppPermissionService.get_user_app_permission(
                app_model.id, current_user.id
            )
            if not user_permission or user_permission.role != AppPermissionRole.OWNER:
                if current_user.current_role != TenantAccountRole.OWNER:
                    raise Forbidden("Only app owners can grant owner permissions.")

        # Grant permission
        permission = AppPermissionService.grant_permission(
            app_model,
            payload.account_id,
            AppPermissionRole(payload.role),
            current_user.id,
        )

        # Get account info
        account = db.session.get(Account, permission.account_id)
        return {
            **permission.__dict__,
            "account": {
                "id": account.id,
                "name": account.name,
                "email": account.email,
                "avatar": account.avatar,
            } if account else None,
        }


@console_ns.route("/apps/<uuid:app_id>/permissions/bulk")
class AppPermissionBulkApi(Resource):
    """Bulk grant permissions."""

    @setup_required
    @login_required
    @account_initialization_required
    @get_app_model
    @marshal_with(app_permission_list_fields)
    def post(self, app_model):
        """Grant permission to multiple users at once."""
        current_user, _ = current_account_with_tenant()

        # Check if user can manage this app
        if not AppPermissionService.can_manage_app(app_model, current_user):
            raise Forbidden("You do not have permission to manage app permissions.")

        payload = BulkGrantPermissionPayload.model_validate(console_ns.payload)

        # Validate role
        if not AppPermissionRole.is_valid_role(payload.role):
            raise ValueError(f"Invalid role: {payload.role}")

        # Only owners can grant owner role
        if payload.role == AppPermissionRole.OWNER:
            raise Forbidden("Cannot bulk grant owner permissions.")

        # Grant permissions
        permissions = AppPermissionService.bulk_grant_permissions(
            app_model,
            payload.account_ids,
            AppPermissionRole(payload.role),
            current_user.id,
        )

        # Enrich with account info
        result = []
        for perm in permissions:
            account = db.session.get(Account, perm.account_id)
            result.append({
                **perm.__dict__,
                "account": {
                    "id": account.id,
                    "name": account.name,
                    "email": account.email,
                    "avatar": account.avatar,
                } if account else None,
            })

        return {
            "permissions": result,
            "total": len(result),
        }


@console_ns.route("/apps/<uuid:app_id>/permissions/<uuid:account_id>")
class AppPermissionDetailApi(Resource):
    """Get, update, or revoke a specific permission."""

    @setup_required
    @login_required
    @account_initialization_required
    @get_app_model
    @marshal_with(app_permission_fields)
    def get(self, app_model, account_id):
        """Get a specific user's permission for an app."""
        current_user, _ = current_account_with_tenant()

        # Users can view their own permission, or managers can view all
        if str(account_id) != current_user.id:
            if not AppPermissionService.can_manage_app(app_model, current_user):
                raise Forbidden("You do not have permission to view this.")

        permission = AppPermissionService.get_user_app_permission(
            app_model.id, str(account_id)
        )
        if not permission:
            raise NotFound("Permission not found.")

        account = db.session.get(Account, permission.account_id)
        return {
            **permission.__dict__,
            "account": {
                "id": account.id,
                "name": account.name,
                "email": account.email,
                "avatar": account.avatar,
            } if account else None,
        }

    @setup_required
    @login_required
    @account_initialization_required
    @get_app_model
    @marshal_with(app_permission_fields)
    def put(self, app_model, account_id):
        """Update a user's permission role."""
        current_user, _ = current_account_with_tenant()

        # Check if user can manage this app
        if not AppPermissionService.can_manage_app(app_model, current_user):
            raise Forbidden("You do not have permission to manage app permissions.")

        payload = GrantPermissionPayload.model_validate(console_ns.payload)

        # Validate role
        if not AppPermissionRole.is_valid_role(payload.role):
            raise ValueError(f"Invalid role: {payload.role}")

        # Only owners can grant owner role
        if payload.role == AppPermissionRole.OWNER:
            user_permission = AppPermissionService.get_user_app_permission(
                app_model.id, current_user.id
            )
            if not user_permission or user_permission.role != AppPermissionRole.OWNER:
                if current_user.current_role != TenantAccountRole.OWNER:
                    raise Forbidden("Only app owners can grant owner permissions.")

        # Grant/update permission
        permission = AppPermissionService.grant_permission(
            app_model,
            str(account_id),
            AppPermissionRole(payload.role),
            current_user.id,
        )

        account = db.session.get(Account, permission.account_id)
        return {
            **permission.__dict__,
            "account": {
                "id": account.id,
                "name": account.name,
                "email": account.email,
                "avatar": account.avatar,
            } if account else None,
        }

    @setup_required
    @login_required
    @account_initialization_required
    @get_app_model
    def delete(self, app_model, account_id):
        """Revoke a user's permission for an app."""
        current_user, _ = current_account_with_tenant()

        # Check if user can manage this app
        if not AppPermissionService.can_manage_app(app_model, current_user):
            raise Forbidden("You do not have permission to manage app permissions.")

        # Cannot revoke your own permission
        if str(account_id) == current_user.id:
            raise Forbidden("You cannot revoke your own permission.")

        # Check if target is owner - only workspace owner can revoke
        target_permission = AppPermissionService.get_user_app_permission(
            app_model.id, str(account_id)
        )
        if target_permission and target_permission.role == AppPermissionRole.OWNER:
            if current_user.current_role != TenantAccountRole.OWNER:
                raise Forbidden("Only workspace owners can revoke app owner permissions.")

        success = AppPermissionService.revoke_permission(app_model.id, str(account_id))
        if not success:
            raise NotFound("Permission not found.")

        return {"message": "Permission revoked successfully."}, 200


@console_ns.route("/apps/<uuid:app_id>/access-config")
class AppAccessConfigApi(Resource):
    """Get and update app access configuration."""

    @setup_required
    @login_required
    @account_initialization_required
    @get_app_model
    @marshal_with(app_access_config_fields)
    def get(self, app_model):
        """Get access configuration for an app."""
        current_user, _ = current_account_with_tenant()

        # Check if user can manage this app
        if not AppPermissionService.can_manage_app(app_model, current_user):
            raise Forbidden("You do not have permission to view access configuration.")

        config = AppPermissionService.get_app_access_config(app_model.id)
        if not config:
            # Return default config
            return {
                "id": None,
                "app_id": app_model.id,
                "tenant_id": app_model.tenant_id,
                "permission_type": AppPermissionType.INHERIT_WORKSPACE,
                "require_api_scope": False,
                "custom_rate_limit_rpm": None,
                "custom_rate_limit_rph": None,
                "access_description": None,
                "created_at": None,
                "updated_at": None,
            }
        return config

    @setup_required
    @login_required
    @account_initialization_required
    @is_admin_or_owner_required
    @get_app_model
    @marshal_with(app_access_config_fields)
    def put(self, app_model):
        """Update access configuration for an app."""
        current_user, _ = current_account_with_tenant()

        payload = UpdateAccessConfigPayload.model_validate(console_ns.payload)

        # Validate permission type
        try:
            permission_type = AppPermissionType(payload.permission_type)
        except ValueError:
            raise ValueError(f"Invalid permission type: {payload.permission_type}")

        # When switching to RESTRICTED, ensure current user gets OWNER permission
        if permission_type == AppPermissionType.RESTRICTED:
            existing_permission = AppPermissionService.get_user_app_permission(
                app_model.id, current_user.id
            )
            if not existing_permission:
                AppPermissionService.grant_permission(
                    app_model,
                    current_user.id,
                    AppPermissionRole.OWNER,
                    current_user.id,
                )

        config = AppPermissionService.create_or_update_access_config(
            app_model,
            permission_type,
            payload.require_api_scope,
            payload.custom_rate_limit_rpm,
            payload.custom_rate_limit_rph,
            payload.access_description,
        )

        return config

    @setup_required
    @login_required
    @account_initialization_required
    @is_admin_or_owner_required
    @get_app_model
    def delete(self, app_model):
        """Delete access configuration (revert to default workspace inheritance)."""
        success = AppPermissionService.delete_access_config(app_model.id)
        if success:
            return {"message": "Access configuration deleted. App now inherits workspace permissions."}
        return {"message": "No custom access configuration found."}


@console_ns.route("/apps/<uuid:app_id>/permissions/check")
class AppPermissionCheckApi(Resource):
    """Check current user's permission for an app."""

    @setup_required
    @login_required
    @account_initialization_required
    @get_app_model
    def get(self, app_model):
        """Check what permissions current user has for this app."""
        current_user, _ = current_account_with_tenant()

        # Get explicit permission
        permission = AppPermissionService.get_user_app_permission(
            app_model.id, current_user.id
        )

        # Get access config
        access_config = AppPermissionService.get_app_access_config(app_model.id)

        return {
            "has_access": AppPermissionService.can_view_app(app_model, current_user),
            "can_invoke": AppPermissionService.can_invoke_app(app_model, current_user),
            "can_manage": AppPermissionService.can_manage_app(app_model, current_user),
            "explicit_role": permission.role if permission else None,
            "permission_source": "explicit" if permission else "workspace",
            "access_config": {
                "permission_type": access_config.permission_type if access_config else "inherit_workspace",
                "require_api_scope": access_config.require_api_scope if access_config else False,
            } if access_config else None,
        }
