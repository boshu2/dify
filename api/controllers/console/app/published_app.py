"""
Published App API Endpoints - Console API for publishing apps externally.

This module provides REST API endpoints for publishing apps for external
consumption, managing versions, and controlling visibility.
"""

from flask_restx import Resource, fields, marshal_with
from pydantic import BaseModel, Field
from werkzeug.exceptions import Forbidden, NotFound

from controllers.console import console_ns
from controllers.console.app.wraps import get_app_model
from controllers.console.wraps import (
    account_initialization_required,
    is_admin_or_owner_required,
    setup_required,
)
from libs.helper import TimestampField
from libs.login import current_account_with_tenant, login_required
from models.published_app import (
    PublishedAppStatus,
    PublishedAppVisibility,
)
from services.published_app_service import (
    PublishedAppAlreadyExistsError,
    PublishedAppNotFoundError,
    PublishedAppService,
)


# ============================================
# Pydantic Request Models
# ============================================

class PublishAppPayload(BaseModel):
    """Payload for publishing an app."""
    name: str = Field(..., min_length=1, max_length=255)
    description: str | None = Field(default=None)
    icon: str | None = Field(default=None, max_length=255)
    icon_background: str | None = Field(default=None, max_length=32)
    version: str = Field(default="1.0.0", max_length=32)
    changelog: str | None = Field(default=None)
    visibility: str = Field(default="private")
    default_rate_limit_rpm: int = Field(default=60, ge=0)
    default_rate_limit_rph: int = Field(default=1000, ge=0)
    default_rate_limit_rpd: int | None = Field(default=None, ge=0)
    free_quota_per_consumer: int | None = Field(default=None, ge=0)
    terms_of_service: str | None = Field(default=None)
    privacy_policy: str | None = Field(default=None)
    documentation_url: str | None = Field(default=None, max_length=512)
    support_email: str | None = Field(default=None, max_length=255)


class UpdatePublishedAppPayload(BaseModel):
    """Payload for updating a published app."""
    name: str | None = Field(default=None, max_length=255)
    description: str | None = Field(default=None)
    icon: str | None = Field(default=None, max_length=255)
    icon_background: str | None = Field(default=None, max_length=32)
    visibility: str | None = Field(default=None)
    default_rate_limit_rpm: int | None = Field(default=None, ge=0)
    default_rate_limit_rph: int | None = Field(default=None, ge=0)
    default_rate_limit_rpd: int | None = Field(default=None, ge=0)
    free_quota_per_consumer: int | None = Field(default=None, ge=0)
    terms_of_service: str | None = Field(default=None)
    privacy_policy: str | None = Field(default=None)
    documentation_url: str | None = Field(default=None, max_length=512)
    support_email: str | None = Field(default=None, max_length=255)
    custom_domain: str | None = Field(default=None, max_length=255)
    cors_allowed_origins: str | None = Field(default=None)
    webhook_url: str | None = Field(default=None, max_length=512)


class CreateVersionPayload(BaseModel):
    """Payload for creating a new version."""
    version: str = Field(..., max_length=32)
    changelog: str | None = Field(default=None)
    workflow_version: str | None = Field(default=None, max_length=255)
    config_snapshot: str | None = Field(default=None)
    is_current: bool = Field(default=True)


class AddWebhookPayload(BaseModel):
    """Payload for adding a webhook."""
    name: str = Field(..., max_length=255)
    url: str = Field(..., max_length=512)
    events: list[str] = Field(...)
    secret: str | None = Field(default=None, max_length=255)


# ============================================
# Flask-RESTX Response Fields
# ============================================

published_app_fields = {
    "id": fields.String,
    "tenant_id": fields.String,
    "app_id": fields.String,
    "slug": fields.String,
    "name": fields.String,
    "description": fields.String,
    "icon": fields.String,
    "icon_background": fields.String,
    "version": fields.String,
    "changelog": fields.String,
    "status": fields.String,
    "visibility": fields.String,
    "default_rate_limit_rpm": fields.Integer,
    "default_rate_limit_rph": fields.Integer,
    "default_rate_limit_rpd": fields.Integer,
    "free_quota_per_consumer": fields.Integer,
    "terms_of_service": fields.String,
    "privacy_policy": fields.String,
    "documentation_url": fields.String,
    "support_email": fields.String,
    "custom_domain": fields.String,
    "cors_allowed_origins": fields.String,
    "webhook_url": fields.String,
    "total_consumers": fields.Integer,
    "total_requests": fields.Integer,
    "published_by": fields.String,
    "published_at": TimestampField,
    "created_by": fields.String,
    "created_at": TimestampField,
    "updated_at": TimestampField,
    "api_base_url": fields.String(attribute="api_base_url"),
}

published_app_list_fields = {
    "apps": fields.List(fields.Nested(published_app_fields)),
    "total": fields.Integer,
    "page": fields.Integer,
    "limit": fields.Integer,
}

version_fields = {
    "id": fields.String,
    "published_app_id": fields.String,
    "version": fields.String,
    "workflow_version": fields.String,
    "config_snapshot": fields.String,
    "changelog": fields.String,
    "is_current": fields.Boolean,
    "is_deprecated": fields.Boolean,
    "deprecation_message": fields.String,
    "created_by": fields.String,
    "created_at": TimestampField,
}

webhook_fields = {
    "id": fields.String,
    "published_app_id": fields.String,
    "tenant_id": fields.String,
    "name": fields.String,
    "url": fields.String,
    "events": fields.String,
    "is_active": fields.Boolean,
    "consecutive_failures": fields.Integer,
    "last_failure_at": TimestampField,
    "last_failure_reason": fields.String,
    "created_at": TimestampField,
    "updated_at": TimestampField,
}


# ============================================
# API Endpoints
# ============================================

@console_ns.route("/published-apps")
class PublishedAppListApi(Resource):
    """List published apps for the workspace."""

    @setup_required
    @login_required
    @account_initialization_required
    @marshal_with(published_app_list_fields)
    def get(self):
        """Get list of all published apps for the workspace."""
        _, tenant_id = current_account_with_tenant()

        page = console_ns.request.args.get("page", 1, type=int)
        limit = console_ns.request.args.get("limit", 20, type=int)
        status = console_ns.request.args.get("status")
        visibility = console_ns.request.args.get("visibility")

        status_filter = None
        if status:
            try:
                status_filter = PublishedAppStatus(status)
            except ValueError:
                pass

        visibility_filter = None
        if visibility:
            try:
                visibility_filter = PublishedAppVisibility(visibility)
            except ValueError:
                pass

        apps, total = PublishedAppService.get_published_apps(
            tenant_id=tenant_id,
            status=status_filter,
            visibility=visibility_filter,
            limit=limit,
            offset=(page - 1) * limit,
        )

        return {
            "apps": apps,
            "total": total,
            "page": page,
            "limit": limit,
        }


@console_ns.route("/apps/<uuid:app_id>/publish")
class AppPublishApi(Resource):
    """Publish an app for external consumption."""

    @setup_required
    @login_required
    @account_initialization_required
    @is_admin_or_owner_required
    @get_app_model
    @marshal_with(published_app_fields)
    def post(self, app_model):
        """Publish an app."""
        current_user, _ = current_account_with_tenant()

        payload = PublishAppPayload.model_validate(console_ns.payload)

        # Validate visibility
        try:
            visibility = PublishedAppVisibility(payload.visibility)
        except ValueError:
            raise ValueError(f"Invalid visibility: {payload.visibility}")

        try:
            published_app = PublishedAppService.publish_app(
                app=app_model,
                name=payload.name,
                created_by=current_user.id,
                description=payload.description,
                icon=payload.icon,
                icon_background=payload.icon_background,
                version=payload.version,
                changelog=payload.changelog,
                visibility=visibility,
                default_rate_limit_rpm=payload.default_rate_limit_rpm,
                default_rate_limit_rph=payload.default_rate_limit_rph,
                default_rate_limit_rpd=payload.default_rate_limit_rpd,
                free_quota_per_consumer=payload.free_quota_per_consumer,
                terms_of_service=payload.terms_of_service,
                privacy_policy=payload.privacy_policy,
                documentation_url=payload.documentation_url,
                support_email=payload.support_email,
            )
            return published_app
        except PublishedAppAlreadyExistsError as e:
            raise Forbidden(str(e))

    @setup_required
    @login_required
    @account_initialization_required
    @get_app_model
    @marshal_with(published_app_fields)
    def get(self, app_model):
        """Get published app info for an internal app."""
        _, tenant_id = current_account_with_tenant()

        published_app = PublishedAppService.get_published_app_by_app_id(
            app_model.id, tenant_id
        )
        if not published_app:
            raise NotFound("This app is not published.")

        return published_app


@console_ns.route("/published-apps/<uuid:published_app_id>")
class PublishedAppDetailApi(Resource):
    """Get, update, or unpublish a published app."""

    @setup_required
    @login_required
    @account_initialization_required
    @marshal_with(published_app_fields)
    def get(self, published_app_id):
        """Get details of a published app."""
        _, tenant_id = current_account_with_tenant()

        published_app = PublishedAppService.get_published_app(str(published_app_id))
        if not published_app or published_app.tenant_id != tenant_id:
            raise NotFound("Published app not found.")

        return published_app

    @setup_required
    @login_required
    @account_initialization_required
    @is_admin_or_owner_required
    @marshal_with(published_app_fields)
    def put(self, published_app_id):
        """Update a published app."""
        _, tenant_id = current_account_with_tenant()

        published_app = PublishedAppService.get_published_app(str(published_app_id))
        if not published_app or published_app.tenant_id != tenant_id:
            raise NotFound("Published app not found.")

        payload = UpdatePublishedAppPayload.model_validate(console_ns.payload)

        visibility = None
        if payload.visibility:
            try:
                visibility = PublishedAppVisibility(payload.visibility)
            except ValueError:
                raise ValueError(f"Invalid visibility: {payload.visibility}")

        published_app = PublishedAppService.update_published_app(
            published_app_id=str(published_app_id),
            name=payload.name,
            description=payload.description,
            icon=payload.icon,
            icon_background=payload.icon_background,
            visibility=visibility,
            default_rate_limit_rpm=payload.default_rate_limit_rpm,
            default_rate_limit_rph=payload.default_rate_limit_rph,
            default_rate_limit_rpd=payload.default_rate_limit_rpd,
            free_quota_per_consumer=payload.free_quota_per_consumer,
            terms_of_service=payload.terms_of_service,
            privacy_policy=payload.privacy_policy,
            documentation_url=payload.documentation_url,
            support_email=payload.support_email,
            custom_domain=payload.custom_domain,
            cors_allowed_origins=payload.cors_allowed_origins,
            webhook_url=payload.webhook_url,
        )

        return published_app

    @setup_required
    @login_required
    @account_initialization_required
    @is_admin_or_owner_required
    def delete(self, published_app_id):
        """Unpublish an app (remove from external access)."""
        _, tenant_id = current_account_with_tenant()

        published_app = PublishedAppService.get_published_app(str(published_app_id))
        if not published_app or published_app.tenant_id != tenant_id:
            raise NotFound("Published app not found.")

        success = PublishedAppService.unpublish(str(published_app_id))
        if success:
            return {"message": "App unpublished successfully."}, 200
        return {"message": "Failed to unpublish app."}, 500


@console_ns.route("/published-apps/<uuid:published_app_id>/status")
class PublishedAppStatusApi(Resource):
    """Manage published app status."""

    @setup_required
    @login_required
    @account_initialization_required
    @is_admin_or_owner_required
    @marshal_with(published_app_fields)
    def post(self, published_app_id):
        """Update published app status."""
        current_user, tenant_id = current_account_with_tenant()

        published_app = PublishedAppService.get_published_app(str(published_app_id))
        if not published_app or published_app.tenant_id != tenant_id:
            raise NotFound("Published app not found.")

        action = console_ns.payload.get("action")

        try:
            if action == "go_live":
                published_app = PublishedAppService.go_live(
                    str(published_app_id), current_user.id
                )
            elif action == "pause":
                published_app = PublishedAppService.pause(str(published_app_id))
            elif action == "deprecate":
                published_app = PublishedAppService.deprecate(str(published_app_id))
            elif action == "archive":
                published_app = PublishedAppService.archive(str(published_app_id))
            else:
                raise ValueError(
                    f"Invalid action: {action}. Use go_live, pause, deprecate, or archive."
                )

            return published_app
        except PublishedAppNotFoundError:
            raise NotFound("Published app not found.")


@console_ns.route("/published-apps/<uuid:published_app_id>/versions")
class PublishedAppVersionListApi(Resource):
    """List and create versions for a published app."""

    @setup_required
    @login_required
    @account_initialization_required
    def get(self, published_app_id):
        """Get all versions for a published app."""
        _, tenant_id = current_account_with_tenant()

        published_app = PublishedAppService.get_published_app(str(published_app_id))
        if not published_app or published_app.tenant_id != tenant_id:
            raise NotFound("Published app not found.")

        versions = PublishedAppService.get_versions(str(published_app_id))

        return {
            "versions": [v.__dict__ for v in versions],
            "total": len(versions),
        }

    @setup_required
    @login_required
    @account_initialization_required
    @is_admin_or_owner_required
    @marshal_with(version_fields)
    def post(self, published_app_id):
        """Create a new version."""
        current_user, tenant_id = current_account_with_tenant()

        published_app = PublishedAppService.get_published_app(str(published_app_id))
        if not published_app or published_app.tenant_id != tenant_id:
            raise NotFound("Published app not found.")

        payload = CreateVersionPayload.model_validate(console_ns.payload)

        version = PublishedAppService.create_version(
            published_app=published_app,
            version=payload.version,
            created_by=current_user.id,
            changelog=payload.changelog,
            workflow_version=payload.workflow_version,
            config_snapshot=payload.config_snapshot,
            is_current=payload.is_current,
        )

        return version


@console_ns.route("/published-apps/<uuid:published_app_id>/versions/<string:version>/current")
class PublishedAppVersionCurrentApi(Resource):
    """Set a version as current."""

    @setup_required
    @login_required
    @account_initialization_required
    @is_admin_or_owner_required
    @marshal_with(version_fields)
    def post(self, published_app_id, version):
        """Set a specific version as current."""
        _, tenant_id = current_account_with_tenant()

        published_app = PublishedAppService.get_published_app(str(published_app_id))
        if not published_app or published_app.tenant_id != tenant_id:
            raise NotFound("Published app not found.")

        try:
            version_record = PublishedAppService.set_current_version(
                str(published_app_id), version
            )
            return version_record
        except PublishedAppNotFoundError as e:
            raise NotFound(str(e))


@console_ns.route("/published-apps/<uuid:published_app_id>/versions/<string:version>/deprecate")
class PublishedAppVersionDeprecateApi(Resource):
    """Deprecate a specific version."""

    @setup_required
    @login_required
    @account_initialization_required
    @is_admin_or_owner_required
    @marshal_with(version_fields)
    def post(self, published_app_id, version):
        """Deprecate a specific version."""
        _, tenant_id = current_account_with_tenant()

        published_app = PublishedAppService.get_published_app(str(published_app_id))
        if not published_app or published_app.tenant_id != tenant_id:
            raise NotFound("Published app not found.")

        message = console_ns.payload.get("message")

        try:
            version_record = PublishedAppService.deprecate_version(
                str(published_app_id), version, message
            )
            return version_record
        except PublishedAppNotFoundError as e:
            raise NotFound(str(e))


@console_ns.route("/published-apps/<uuid:published_app_id>/webhooks")
class PublishedAppWebhookListApi(Resource):
    """List and add webhooks for a published app."""

    @setup_required
    @login_required
    @account_initialization_required
    def get(self, published_app_id):
        """Get all webhooks for a published app."""
        _, tenant_id = current_account_with_tenant()

        published_app = PublishedAppService.get_published_app(str(published_app_id))
        if not published_app or published_app.tenant_id != tenant_id:
            raise NotFound("Published app not found.")

        webhooks = PublishedAppService.get_webhooks(str(published_app_id))

        return {
            "webhooks": [w.__dict__ for w in webhooks],
            "total": len(webhooks),
        }

    @setup_required
    @login_required
    @account_initialization_required
    @is_admin_or_owner_required
    @marshal_with(webhook_fields)
    def post(self, published_app_id):
        """Add a webhook."""
        _, tenant_id = current_account_with_tenant()

        published_app = PublishedAppService.get_published_app(str(published_app_id))
        if not published_app or published_app.tenant_id != tenant_id:
            raise NotFound("Published app not found.")

        payload = AddWebhookPayload.model_validate(console_ns.payload)

        webhook = PublishedAppService.add_webhook(
            published_app_id=str(published_app_id),
            tenant_id=tenant_id,
            name=payload.name,
            url=payload.url,
            events=payload.events,
            secret=payload.secret,
        )

        return webhook


@console_ns.route("/published-apps/<uuid:published_app_id>/webhooks/<uuid:webhook_id>")
class PublishedAppWebhookDetailApi(Resource):
    """Delete a webhook."""

    @setup_required
    @login_required
    @account_initialization_required
    @is_admin_or_owner_required
    def delete(self, published_app_id, webhook_id):
        """Delete a webhook."""
        _, tenant_id = current_account_with_tenant()

        published_app = PublishedAppService.get_published_app(str(published_app_id))
        if not published_app or published_app.tenant_id != tenant_id:
            raise NotFound("Published app not found.")

        success = PublishedAppService.delete_webhook(str(webhook_id))
        if success:
            return {"message": "Webhook deleted successfully."}, 200
        raise NotFound("Webhook not found.")


@console_ns.route("/public/apps")
class PublicAppsApi(Resource):
    """List publicly available apps."""

    def get(self):
        """Get list of publicly discoverable apps."""
        page = console_ns.request.args.get("page", 1, type=int)
        limit = console_ns.request.args.get("limit", 20, type=int)

        apps = PublishedAppService.get_public_apps(
            limit=limit,
            offset=(page - 1) * limit,
        )

        return {
            "apps": [
                {
                    "slug": app.slug,
                    "name": app.name,
                    "description": app.description,
                    "icon": app.icon,
                    "icon_background": app.icon_background,
                    "total_consumers": app.total_consumers,
                    "total_requests": app.total_requests,
                    "api_base_url": app.api_base_url,
                }
                for app in apps
            ],
            "page": page,
            "limit": limit,
        }


@console_ns.route("/public/apps/<string:slug>")
class PublicAppDetailApi(Resource):
    """Get public app details by slug."""

    def get(self, slug):
        """Get public app details."""
        published_app = PublishedAppService.get_published_app_by_slug(slug)

        if not published_app:
            raise NotFound("App not found.")

        if not published_app.is_live:
            raise NotFound("App is not available.")

        return {
            "slug": published_app.slug,
            "name": published_app.name,
            "description": published_app.description,
            "icon": published_app.icon,
            "icon_background": published_app.icon_background,
            "version": published_app.version,
            "terms_of_service": published_app.terms_of_service,
            "privacy_policy": published_app.privacy_policy,
            "documentation_url": published_app.documentation_url,
            "support_email": published_app.support_email,
            "default_rate_limit_rpm": published_app.default_rate_limit_rpm,
            "default_rate_limit_rph": published_app.default_rate_limit_rph,
            "api_base_url": published_app.api_base_url,
        }
