"""
External Consumer API Endpoints - Console API for managing external API consumers.

This module provides REST API endpoints for managing external consumers,
their authentication, app access grants, and usage tracking.
"""

from datetime import datetime

from flask_restx import Resource, fields, marshal_with
from pydantic import BaseModel, Field
from werkzeug.exceptions import Forbidden, NotFound

from controllers.console import console_ns
from controllers.console.wraps import (
    account_initialization_required,
    is_admin_or_owner_required,
    setup_required,
)
from libs.helper import TimestampField
from libs.login import current_account_with_tenant, login_required
from models.external_consumer import (
    ExternalConsumer,
    ExternalConsumerAuthType,
    ExternalConsumerStatus,
)
from services.external_consumer_service import (
    ExternalConsumerAccessDeniedError,
    ExternalConsumerNotFoundError,
    ExternalConsumerService,
)


# ============================================
# Pydantic Request Models
# ============================================

class CreateConsumerPayload(BaseModel):
    """Payload for creating an external consumer."""
    name: str = Field(..., min_length=1, max_length=255)
    email: str = Field(..., min_length=1, max_length=255)
    organization: str | None = Field(default=None, max_length=255)
    description: str | None = Field(default=None)
    auth_type: str = Field(default="api_key")
    rate_limit_rpm: int | None = Field(default=None, ge=0)
    rate_limit_rph: int | None = Field(default=None, ge=0)
    rate_limit_rpd: int | None = Field(default=None, ge=0)
    quota_total: int | None = Field(default=None, ge=0)


class UpdateConsumerPayload(BaseModel):
    """Payload for updating an external consumer."""
    name: str | None = Field(default=None, max_length=255)
    organization: str | None = Field(default=None, max_length=255)
    description: str | None = Field(default=None)
    rate_limit_rpm: int | None = Field(default=None, ge=0)
    rate_limit_rph: int | None = Field(default=None, ge=0)
    rate_limit_rpd: int | None = Field(default=None, ge=0)
    quota_total: int | None = Field(default=None, ge=0)


class GrantAppAccessPayload(BaseModel):
    """Payload for granting app access to a consumer."""
    app_id: str = Field(...)
    can_invoke: bool = Field(default=True)
    can_view_logs: bool = Field(default=False)
    custom_rate_limit_rpm: int | None = Field(default=None, ge=0)
    custom_rate_limit_rph: int | None = Field(default=None, ge=0)
    app_quota_total: int | None = Field(default=None, ge=0)
    allowed_scopes: list[str] | None = Field(default=None)
    valid_from: datetime | None = Field(default=None)
    valid_until: datetime | None = Field(default=None)


# ============================================
# Flask-RESTX Response Fields
# ============================================

external_consumer_fields = {
    "id": fields.String,
    "tenant_id": fields.String,
    "name": fields.String,
    "email": fields.String,
    "organization": fields.String,
    "description": fields.String,
    "auth_type": fields.String,
    "api_key_prefix": fields.String,
    "status": fields.String,
    "rate_limit_rpm": fields.Integer,
    "rate_limit_rph": fields.Integer,
    "rate_limit_rpd": fields.Integer,
    "quota_total": fields.Integer,
    "quota_used": fields.Integer,
    "quota_reset_at": TimestampField,
    "last_active_at": TimestampField,
    "created_at": TimestampField,
    "updated_at": TimestampField,
}

external_consumer_list_fields = {
    "consumers": fields.List(fields.Nested(external_consumer_fields)),
    "total": fields.Integer,
    "page": fields.Integer,
    "limit": fields.Integer,
}

external_consumer_with_key_fields = {
    **external_consumer_fields,
    "api_key": fields.String,
}

app_access_fields = {
    "id": fields.String,
    "consumer_id": fields.String,
    "app_id": fields.String,
    "tenant_id": fields.String,
    "can_invoke": fields.Boolean,
    "can_view_logs": fields.Boolean,
    "custom_rate_limit_rpm": fields.Integer,
    "custom_rate_limit_rph": fields.Integer,
    "app_quota_total": fields.Integer,
    "app_quota_used": fields.Integer,
    "allowed_scopes": fields.String,
    "valid_from": TimestampField,
    "valid_until": TimestampField,
    "granted_by": fields.String,
    "created_at": TimestampField,
    "updated_at": TimestampField,
}

usage_log_fields = {
    "id": fields.String,
    "consumer_id": fields.String,
    "app_id": fields.String,
    "endpoint": fields.String,
    "method": fields.String,
    "status_code": fields.Integer,
    "response_time_ms": fields.Integer,
    "prompt_tokens": fields.Integer,
    "completion_tokens": fields.Integer,
    "total_tokens": fields.Integer,
    "estimated_cost": fields.Float,
    "request_id": fields.String,
    "ip_address": fields.String,
    "created_at": TimestampField,
}

usage_stats_fields = {
    "total_requests": fields.Integer,
    "total_tokens": fields.Integer,
    "total_cost": fields.Float,
    "avg_response_time": fields.Float,
}


# ============================================
# API Endpoints
# ============================================

@console_ns.route("/external-consumers")
class ExternalConsumerListApi(Resource):
    """List and create external consumers."""

    @setup_required
    @login_required
    @account_initialization_required
    @is_admin_or_owner_required
    @marshal_with(external_consumer_list_fields)
    def get(self):
        """Get list of all external consumers for the workspace."""
        _, tenant_id = current_account_with_tenant()

        page = console_ns.request.args.get("page", 1, type=int)
        limit = console_ns.request.args.get("limit", 20, type=int)
        status = console_ns.request.args.get("status")

        status_filter = None
        if status:
            try:
                status_filter = ExternalConsumerStatus(status)
            except ValueError:
                pass

        consumers, total = ExternalConsumerService.get_consumers(
            tenant_id=tenant_id,
            status=status_filter,
            limit=limit,
            offset=(page - 1) * limit,
        )

        return {
            "consumers": consumers,
            "total": total,
            "page": page,
            "limit": limit,
        }

    @setup_required
    @login_required
    @account_initialization_required
    @is_admin_or_owner_required
    @marshal_with(external_consumer_with_key_fields)
    def post(self):
        """Create a new external consumer."""
        current_user, tenant_id = current_account_with_tenant()

        payload = CreateConsumerPayload.model_validate(console_ns.payload)

        # Validate auth type
        try:
            auth_type = ExternalConsumerAuthType(payload.auth_type)
        except ValueError:
            raise ValueError(f"Invalid auth type: {payload.auth_type}")

        # Check for existing consumer with same email
        existing = ExternalConsumerService.get_consumer_by_email(
            tenant_id, payload.email
        )
        if existing:
            raise ValueError(f"Consumer with email {payload.email} already exists.")

        consumer, api_key = ExternalConsumerService.create_consumer(
            tenant_id=tenant_id,
            name=payload.name,
            email=payload.email,
            created_by=current_user.id,
            organization=payload.organization,
            description=payload.description,
            auth_type=auth_type,
            rate_limit_rpm=payload.rate_limit_rpm,
            rate_limit_rph=payload.rate_limit_rph,
            rate_limit_rpd=payload.rate_limit_rpd,
            quota_total=payload.quota_total,
        )

        return {
            **consumer.__dict__,
            "api_key": api_key,
        }


@console_ns.route("/external-consumers/<uuid:consumer_id>")
class ExternalConsumerDetailApi(Resource):
    """Get, update, or delete a specific external consumer."""

    @setup_required
    @login_required
    @account_initialization_required
    @is_admin_or_owner_required
    @marshal_with(external_consumer_fields)
    def get(self, consumer_id):
        """Get details of a specific consumer."""
        _, tenant_id = current_account_with_tenant()

        consumer = ExternalConsumerService.get_consumer(str(consumer_id))
        if not consumer or consumer.tenant_id != tenant_id:
            raise NotFound("Consumer not found.")

        return consumer

    @setup_required
    @login_required
    @account_initialization_required
    @is_admin_or_owner_required
    @marshal_with(external_consumer_fields)
    def put(self, consumer_id):
        """Update a consumer's details."""
        _, tenant_id = current_account_with_tenant()

        consumer = ExternalConsumerService.get_consumer(str(consumer_id))
        if not consumer or consumer.tenant_id != tenant_id:
            raise NotFound("Consumer not found.")

        payload = UpdateConsumerPayload.model_validate(console_ns.payload)

        consumer = ExternalConsumerService.update_consumer(
            consumer_id=str(consumer_id),
            name=payload.name,
            organization=payload.organization,
            description=payload.description,
            rate_limit_rpm=payload.rate_limit_rpm,
            rate_limit_rph=payload.rate_limit_rph,
            rate_limit_rpd=payload.rate_limit_rpd,
            quota_total=payload.quota_total,
        )

        return consumer

    @setup_required
    @login_required
    @account_initialization_required
    @is_admin_or_owner_required
    def delete(self, consumer_id):
        """Delete a consumer."""
        _, tenant_id = current_account_with_tenant()

        consumer = ExternalConsumerService.get_consumer(str(consumer_id))
        if not consumer or consumer.tenant_id != tenant_id:
            raise NotFound("Consumer not found.")

        success = ExternalConsumerService.delete_consumer(str(consumer_id))
        if success:
            return {"message": "Consumer deleted successfully."}, 200
        return {"message": "Failed to delete consumer."}, 500


@console_ns.route("/external-consumers/<uuid:consumer_id>/status")
class ExternalConsumerStatusApi(Resource):
    """Manage consumer status (activate, suspend, revoke)."""

    @setup_required
    @login_required
    @account_initialization_required
    @is_admin_or_owner_required
    @marshal_with(external_consumer_fields)
    def post(self, consumer_id):
        """Update consumer status."""
        _, tenant_id = current_account_with_tenant()

        consumer = ExternalConsumerService.get_consumer(str(consumer_id))
        if not consumer or consumer.tenant_id != tenant_id:
            raise NotFound("Consumer not found.")

        action = console_ns.payload.get("action")

        if action == "activate":
            consumer = ExternalConsumerService.activate_consumer(str(consumer_id))
        elif action == "suspend":
            consumer = ExternalConsumerService.suspend_consumer(str(consumer_id))
        elif action == "revoke":
            consumer = ExternalConsumerService.revoke_consumer(str(consumer_id))
        else:
            raise ValueError(f"Invalid action: {action}. Use activate, suspend, or revoke.")

        return consumer


@console_ns.route("/external-consumers/<uuid:consumer_id>/regenerate-key")
class ExternalConsumerRegenerateKeyApi(Resource):
    """Regenerate API key for a consumer."""

    @setup_required
    @login_required
    @account_initialization_required
    @is_admin_or_owner_required
    @marshal_with(external_consumer_with_key_fields)
    def post(self, consumer_id):
        """Regenerate API key for a consumer."""
        _, tenant_id = current_account_with_tenant()

        consumer = ExternalConsumerService.get_consumer(str(consumer_id))
        if not consumer or consumer.tenant_id != tenant_id:
            raise NotFound("Consumer not found.")

        try:
            consumer, api_key = ExternalConsumerService.regenerate_api_key(
                str(consumer_id)
            )
            return {
                **consumer.__dict__,
                "api_key": api_key,
            }
        except ValueError as e:
            raise Forbidden(str(e))


@console_ns.route("/external-consumers/<uuid:consumer_id>/app-access")
class ExternalConsumerAppAccessListApi(Resource):
    """List and grant app access for a consumer."""

    @setup_required
    @login_required
    @account_initialization_required
    @is_admin_or_owner_required
    def get(self, consumer_id):
        """Get all app access grants for a consumer."""
        _, tenant_id = current_account_with_tenant()

        consumer = ExternalConsumerService.get_consumer(str(consumer_id))
        if not consumer or consumer.tenant_id != tenant_id:
            raise NotFound("Consumer not found.")

        accesses = ExternalConsumerService.get_consumer_app_accesses(str(consumer_id))

        return {
            "accesses": [a.__dict__ for a in accesses],
            "total": len(accesses),
        }

    @setup_required
    @login_required
    @account_initialization_required
    @is_admin_or_owner_required
    @marshal_with(app_access_fields)
    def post(self, consumer_id):
        """Grant app access to a consumer."""
        current_user, tenant_id = current_account_with_tenant()

        consumer = ExternalConsumerService.get_consumer(str(consumer_id))
        if not consumer or consumer.tenant_id != tenant_id:
            raise NotFound("Consumer not found.")

        payload = GrantAppAccessPayload.model_validate(console_ns.payload)

        access = ExternalConsumerService.grant_app_access(
            consumer_id=str(consumer_id),
            app_id=payload.app_id,
            tenant_id=tenant_id,
            granted_by=current_user.id,
            can_invoke=payload.can_invoke,
            can_view_logs=payload.can_view_logs,
            custom_rate_limit_rpm=payload.custom_rate_limit_rpm,
            custom_rate_limit_rph=payload.custom_rate_limit_rph,
            app_quota_total=payload.app_quota_total,
            allowed_scopes=payload.allowed_scopes,
            valid_from=payload.valid_from,
            valid_until=payload.valid_until,
        )

        return access


@console_ns.route("/external-consumers/<uuid:consumer_id>/app-access/<uuid:app_id>")
class ExternalConsumerAppAccessDetailApi(Resource):
    """Get or revoke specific app access for a consumer."""

    @setup_required
    @login_required
    @account_initialization_required
    @is_admin_or_owner_required
    @marshal_with(app_access_fields)
    def get(self, consumer_id, app_id):
        """Get a specific app access grant."""
        _, tenant_id = current_account_with_tenant()

        consumer = ExternalConsumerService.get_consumer(str(consumer_id))
        if not consumer or consumer.tenant_id != tenant_id:
            raise NotFound("Consumer not found.")

        access = ExternalConsumerService.get_consumer_app_access(
            str(consumer_id), str(app_id)
        )
        if not access:
            raise NotFound("App access not found.")

        return access

    @setup_required
    @login_required
    @account_initialization_required
    @is_admin_or_owner_required
    def delete(self, consumer_id, app_id):
        """Revoke app access from a consumer."""
        _, tenant_id = current_account_with_tenant()

        consumer = ExternalConsumerService.get_consumer(str(consumer_id))
        if not consumer or consumer.tenant_id != tenant_id:
            raise NotFound("Consumer not found.")

        success = ExternalConsumerService.revoke_app_access(
            str(consumer_id), str(app_id)
        )
        if success:
            return {"message": "App access revoked successfully."}, 200
        raise NotFound("App access not found.")


@console_ns.route("/external-consumers/<uuid:consumer_id>/usage")
class ExternalConsumerUsageApi(Resource):
    """Get usage logs and stats for a consumer."""

    @setup_required
    @login_required
    @account_initialization_required
    @is_admin_or_owner_required
    def get(self, consumer_id):
        """Get usage logs for a consumer."""
        _, tenant_id = current_account_with_tenant()

        consumer = ExternalConsumerService.get_consumer(str(consumer_id))
        if not consumer or consumer.tenant_id != tenant_id:
            raise NotFound("Consumer not found.")

        app_id = console_ns.request.args.get("app_id")
        start_date = console_ns.request.args.get("start_date")
        end_date = console_ns.request.args.get("end_date")
        limit = console_ns.request.args.get("limit", 100, type=int)
        offset = console_ns.request.args.get("offset", 0, type=int)

        # Parse dates
        start_dt = datetime.fromisoformat(start_date) if start_date else None
        end_dt = datetime.fromisoformat(end_date) if end_date else None

        logs = ExternalConsumerService.get_consumer_usage_logs(
            consumer_id=str(consumer_id),
            app_id=app_id,
            start_date=start_dt,
            end_date=end_dt,
            limit=limit,
            offset=offset,
        )

        return {
            "logs": [log.__dict__ for log in logs],
            "count": len(logs),
        }


@console_ns.route("/external-consumers/<uuid:consumer_id>/usage/stats")
class ExternalConsumerUsageStatsApi(Resource):
    """Get aggregated usage stats for a consumer."""

    @setup_required
    @login_required
    @account_initialization_required
    @is_admin_or_owner_required
    @marshal_with(usage_stats_fields)
    def get(self, consumer_id):
        """Get aggregated usage stats for a consumer."""
        _, tenant_id = current_account_with_tenant()

        consumer = ExternalConsumerService.get_consumer(str(consumer_id))
        if not consumer or consumer.tenant_id != tenant_id:
            raise NotFound("Consumer not found.")

        app_id = console_ns.request.args.get("app_id")
        start_date = console_ns.request.args.get("start_date")
        end_date = console_ns.request.args.get("end_date")

        # Parse dates
        start_dt = datetime.fromisoformat(start_date) if start_date else None
        end_dt = datetime.fromisoformat(end_date) if end_date else None

        stats = ExternalConsumerService.get_consumer_usage_stats(
            consumer_id=str(consumer_id),
            app_id=app_id,
            start_date=start_dt,
            end_date=end_dt,
        )

        return stats


@console_ns.route("/external-consumers/<uuid:consumer_id>/reset-quota")
class ExternalConsumerResetQuotaApi(Resource):
    """Reset quota for a consumer."""

    @setup_required
    @login_required
    @account_initialization_required
    @is_admin_or_owner_required
    @marshal_with(external_consumer_fields)
    def post(self, consumer_id):
        """Reset consumer's quota usage."""
        _, tenant_id = current_account_with_tenant()

        consumer = ExternalConsumerService.get_consumer(str(consumer_id))
        if not consumer or consumer.tenant_id != tenant_id:
            raise NotFound("Consumer not found.")

        try:
            consumer = ExternalConsumerService.reset_consumer_quota(str(consumer_id))
            return consumer
        except ExternalConsumerNotFoundError:
            raise NotFound("Consumer not found.")


@console_ns.route("/apps/<uuid:app_id>/external-consumers")
class AppExternalConsumersApi(Resource):
    """Get consumers with access to a specific app."""

    @setup_required
    @login_required
    @account_initialization_required
    @is_admin_or_owner_required
    def get(self, app_id):
        """Get all consumers with access to this app."""
        _, tenant_id = current_account_with_tenant()

        accesses = ExternalConsumerService.get_app_consumers(str(app_id))

        # Filter by tenant
        accesses = [a for a in accesses if a.tenant_id == tenant_id]

        # Enrich with consumer info
        result = []
        for access in accesses:
            consumer = ExternalConsumerService.get_consumer(access.consumer_id)
            if consumer:
                result.append({
                    "access": access.__dict__,
                    "consumer": {
                        "id": consumer.id,
                        "name": consumer.name,
                        "email": consumer.email,
                        "organization": consumer.organization,
                        "status": consumer.status,
                    },
                })

        return {
            "consumers": result,
            "total": len(result),
        }
