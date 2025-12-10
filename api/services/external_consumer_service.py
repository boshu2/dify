"""
External Consumer Service - Business logic for external API consumers.

This service provides methods for managing external consumers,
their authentication, app access grants, and usage tracking.
"""

from collections.abc import Sequence
from datetime import datetime

from sqlalchemy import select

from extensions.ext_database import db
from models.external_consumer import (
    ExternalConsumer,
    ExternalConsumerAppAccess,
    ExternalConsumerAuthType,
    ExternalConsumerStatus,
    ExternalConsumerUsageLog,
)


class ExternalConsumerNotFoundError(Exception):
    """Raised when an external consumer is not found."""

    def __init__(self, message: str = "External consumer not found."):
        self.message = message
        super().__init__(self.message)


class ExternalConsumerAccessDeniedError(Exception):
    """Raised when an external consumer is denied access."""

    def __init__(self, message: str = "Access denied."):
        self.message = message
        super().__init__(self.message)


class ExternalConsumerQuotaExceededError(Exception):
    """Raised when an external consumer exceeds their quota."""

    def __init__(self, message: str = "Quota exceeded."):
        self.message = message
        super().__init__(self.message)


class ExternalConsumerService:
    """Service for managing external API consumers."""

    # ==========================================
    # Consumer CRUD Methods
    # ==========================================

    @classmethod
    def create_consumer(
        cls,
        tenant_id: str,
        name: str,
        email: str,
        created_by: str,
        organization: str | None = None,
        description: str | None = None,
        auth_type: ExternalConsumerAuthType = ExternalConsumerAuthType.API_KEY,
        rate_limit_rpm: int | None = None,
        rate_limit_rph: int | None = None,
        rate_limit_rpd: int | None = None,
        quota_total: int | None = None,
    ) -> tuple[ExternalConsumer, str | None]:
        """
        Create a new external consumer.

        Args:
            tenant_id: The workspace ID
            name: Consumer name
            email: Consumer email
            created_by: ID of the user creating the consumer
            organization: Optional organization name
            description: Optional description
            auth_type: Authentication type (api_key, oauth, jwt)
            rate_limit_rpm: Requests per minute limit
            rate_limit_rph: Requests per hour limit
            rate_limit_rpd: Requests per day limit
            quota_total: Total quota limit

        Returns:
            Tuple of (consumer, api_key) - api_key is None if auth_type is not API_KEY
        """
        api_key = None
        api_key_hash = None
        api_key_prefix = None

        # Generate API key if using API key auth
        if auth_type == ExternalConsumerAuthType.API_KEY:
            api_key, api_key_prefix, api_key_hash = ExternalConsumer.generate_api_key()

        consumer = ExternalConsumer(
            tenant_id=tenant_id,
            name=name,
            email=email,
            organization=organization,
            description=description,
            auth_type=auth_type.value,
            api_key_hash=api_key_hash,
            api_key_prefix=api_key_prefix,
            rate_limit_rpm=rate_limit_rpm,
            rate_limit_rph=rate_limit_rph,
            rate_limit_rpd=rate_limit_rpd,
            quota_total=quota_total,
            created_by=created_by,
        )

        db.session.add(consumer)
        db.session.commit()

        return consumer, api_key

    @classmethod
    def get_consumer(cls, consumer_id: str) -> ExternalConsumer | None:
        """Get a consumer by ID."""
        return db.session.get(ExternalConsumer, consumer_id)

    @classmethod
    def get_consumer_by_email(cls, tenant_id: str, email: str) -> ExternalConsumer | None:
        """Get a consumer by email within a tenant."""
        return db.session.scalar(
            select(ExternalConsumer).where(
                ExternalConsumer.tenant_id == tenant_id,
                ExternalConsumer.email == email,
            )
        )

    @classmethod
    def get_consumers(
        cls,
        tenant_id: str,
        status: ExternalConsumerStatus | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[Sequence[ExternalConsumer], int]:
        """
        Get consumers for a tenant with optional filtering.

        Returns:
            Tuple of (consumers, total_count)
        """
        query = select(ExternalConsumer).where(ExternalConsumer.tenant_id == tenant_id)

        if status:
            query = query.where(ExternalConsumer.status == status.value)

        # Get total count
        count_query = select(ExternalConsumer.id).where(ExternalConsumer.tenant_id == tenant_id)
        if status:
            count_query = count_query.where(ExternalConsumer.status == status.value)
        total = len(db.session.scalars(count_query).all())

        # Get paginated results
        query = query.order_by(ExternalConsumer.created_at.desc()).offset(offset).limit(limit)
        consumers = db.session.scalars(query).all()

        return consumers, total

    @classmethod
    def update_consumer(
        cls,
        consumer_id: str,
        name: str | None = None,
        organization: str | None = None,
        description: str | None = None,
        rate_limit_rpm: int | None = None,
        rate_limit_rph: int | None = None,
        rate_limit_rpd: int | None = None,
        quota_total: int | None = None,
    ) -> ExternalConsumer:
        """Update consumer details."""
        consumer = cls.get_consumer(consumer_id)
        if not consumer:
            raise ExternalConsumerNotFoundError()

        if name is not None:
            consumer.name = name
        if organization is not None:
            consumer.organization = organization
        if description is not None:
            consumer.description = description
        if rate_limit_rpm is not None:
            consumer.rate_limit_rpm = rate_limit_rpm
        if rate_limit_rph is not None:
            consumer.rate_limit_rph = rate_limit_rph
        if rate_limit_rpd is not None:
            consumer.rate_limit_rpd = rate_limit_rpd
        if quota_total is not None:
            consumer.quota_total = quota_total

        db.session.commit()
        return consumer

    @classmethod
    def suspend_consumer(cls, consumer_id: str) -> ExternalConsumer:
        """Suspend a consumer (temporarily disable access)."""
        consumer = cls.get_consumer(consumer_id)
        if not consumer:
            raise ExternalConsumerNotFoundError()

        consumer.status = ExternalConsumerStatus.SUSPENDED.value
        db.session.commit()
        return consumer

    @classmethod
    def activate_consumer(cls, consumer_id: str) -> ExternalConsumer:
        """Activate a consumer (enable access)."""
        consumer = cls.get_consumer(consumer_id)
        if not consumer:
            raise ExternalConsumerNotFoundError()

        consumer.status = ExternalConsumerStatus.ACTIVE.value
        db.session.commit()
        return consumer

    @classmethod
    def revoke_consumer(cls, consumer_id: str) -> ExternalConsumer:
        """Permanently revoke a consumer's access."""
        consumer = cls.get_consumer(consumer_id)
        if not consumer:
            raise ExternalConsumerNotFoundError()

        consumer.status = ExternalConsumerStatus.REVOKED.value
        db.session.commit()
        return consumer

    @classmethod
    def delete_consumer(cls, consumer_id: str) -> bool:
        """Delete a consumer and all their app access grants."""
        consumer = cls.get_consumer(consumer_id)
        if not consumer:
            return False

        # Delete app access grants first
        db.session.execute(
            ExternalConsumerAppAccess.__table__.delete().where(
                ExternalConsumerAppAccess.consumer_id == consumer_id
            )
        )

        db.session.delete(consumer)
        db.session.commit()
        return True

    # ==========================================
    # API Key Management
    # ==========================================

    @classmethod
    def regenerate_api_key(cls, consumer_id: str) -> tuple[ExternalConsumer, str]:
        """
        Regenerate API key for a consumer.

        Returns:
            Tuple of (consumer, new_api_key)
        """
        consumer = cls.get_consumer(consumer_id)
        if not consumer:
            raise ExternalConsumerNotFoundError()

        if consumer.auth_type != ExternalConsumerAuthType.API_KEY.value:
            raise ValueError("Consumer does not use API key authentication.")

        api_key, api_key_prefix, api_key_hash = ExternalConsumer.generate_api_key()
        consumer.api_key_hash = api_key_hash
        consumer.api_key_prefix = api_key_prefix
        db.session.commit()

        return consumer, api_key

    @classmethod
    def authenticate_by_api_key(cls, tenant_id: str, api_key: str) -> ExternalConsumer:
        """
        Authenticate a consumer by API key.

        Args:
            tenant_id: The workspace ID
            api_key: The API key to authenticate

        Returns:
            The authenticated consumer

        Raises:
            ExternalConsumerAccessDeniedError: If authentication fails
        """
        # Get prefix from key
        prefix = api_key[:12] if len(api_key) >= 12 else api_key

        # Find consumer by prefix
        consumer = db.session.scalar(
            select(ExternalConsumer).where(
                ExternalConsumer.tenant_id == tenant_id,
                ExternalConsumer.api_key_prefix == prefix,
                ExternalConsumer.auth_type == ExternalConsumerAuthType.API_KEY.value,
            )
        )

        if not consumer:
            raise ExternalConsumerAccessDeniedError("Invalid API key.")

        # Verify full key
        if not ExternalConsumer.verify_api_key(api_key, consumer.api_key_hash):
            raise ExternalConsumerAccessDeniedError("Invalid API key.")

        # Check status
        if not consumer.is_active:
            raise ExternalConsumerAccessDeniedError(
                f"Consumer is {consumer.status}. Access denied."
            )

        # Update last active timestamp
        consumer.last_active_at = datetime.utcnow()
        db.session.commit()

        return consumer

    # ==========================================
    # App Access Management
    # ==========================================

    @classmethod
    def grant_app_access(
        cls,
        consumer_id: str,
        app_id: str,
        tenant_id: str,
        granted_by: str,
        can_invoke: bool = True,
        can_view_logs: bool = False,
        custom_rate_limit_rpm: int | None = None,
        custom_rate_limit_rph: int | None = None,
        app_quota_total: int | None = None,
        allowed_scopes: list[str] | None = None,
        valid_from: datetime | None = None,
        valid_until: datetime | None = None,
    ) -> ExternalConsumerAppAccess:
        """
        Grant a consumer access to an app.

        Args:
            consumer_id: The consumer ID
            app_id: The app ID to grant access to
            tenant_id: The workspace ID
            granted_by: ID of the user granting access
            can_invoke: Whether consumer can invoke the app
            can_view_logs: Whether consumer can view their logs
            custom_rate_limit_rpm: Custom requests per minute limit for this app
            custom_rate_limit_rph: Custom requests per hour limit for this app
            app_quota_total: Total quota for this specific app
            allowed_scopes: List of allowed API scopes
            valid_from: Access start date
            valid_until: Access end date

        Returns:
            The created or updated AppAccess record
        """
        import json

        # Check for existing access
        existing = cls.get_consumer_app_access(consumer_id, app_id)

        if existing:
            existing.can_invoke = can_invoke
            existing.can_view_logs = can_view_logs
            existing.custom_rate_limit_rpm = custom_rate_limit_rpm
            existing.custom_rate_limit_rph = custom_rate_limit_rph
            existing.app_quota_total = app_quota_total
            existing.allowed_scopes = json.dumps(allowed_scopes) if allowed_scopes else None
            existing.valid_from = valid_from
            existing.valid_until = valid_until
            existing.granted_by = granted_by
            db.session.commit()
            return existing

        access = ExternalConsumerAppAccess(
            consumer_id=consumer_id,
            app_id=app_id,
            tenant_id=tenant_id,
            can_invoke=can_invoke,
            can_view_logs=can_view_logs,
            custom_rate_limit_rpm=custom_rate_limit_rpm,
            custom_rate_limit_rph=custom_rate_limit_rph,
            app_quota_total=app_quota_total,
            allowed_scopes=json.dumps(allowed_scopes) if allowed_scopes else None,
            valid_from=valid_from,
            valid_until=valid_until,
            granted_by=granted_by,
        )
        db.session.add(access)
        db.session.commit()
        return access

    @classmethod
    def get_consumer_app_access(
        cls, consumer_id: str, app_id: str
    ) -> ExternalConsumerAppAccess | None:
        """Get a consumer's access record for a specific app."""
        return db.session.scalar(
            select(ExternalConsumerAppAccess).where(
                ExternalConsumerAppAccess.consumer_id == consumer_id,
                ExternalConsumerAppAccess.app_id == app_id,
            )
        )

    @classmethod
    def get_consumer_app_accesses(
        cls, consumer_id: str
    ) -> Sequence[ExternalConsumerAppAccess]:
        """Get all app access grants for a consumer."""
        return db.session.scalars(
            select(ExternalConsumerAppAccess).where(
                ExternalConsumerAppAccess.consumer_id == consumer_id
            )
        ).all()

    @classmethod
    def get_app_consumers(cls, app_id: str) -> Sequence[ExternalConsumerAppAccess]:
        """Get all consumers with access to an app."""
        return db.session.scalars(
            select(ExternalConsumerAppAccess).where(
                ExternalConsumerAppAccess.app_id == app_id
            )
        ).all()

    @classmethod
    def revoke_app_access(cls, consumer_id: str, app_id: str) -> bool:
        """Revoke a consumer's access to an app."""
        access = cls.get_consumer_app_access(consumer_id, app_id)
        if not access:
            return False

        db.session.delete(access)
        db.session.commit()
        return True

    @classmethod
    def check_app_access(
        cls,
        consumer: ExternalConsumer,
        app_id: str,
        required_scope: str | None = None,
    ) -> ExternalConsumerAppAccess:
        """
        Check if a consumer has access to an app.

        Args:
            consumer: The consumer to check
            app_id: The app ID
            required_scope: Optional specific scope required

        Returns:
            The access record

        Raises:
            ExternalConsumerAccessDeniedError: If access is denied
            ExternalConsumerQuotaExceededError: If quota is exceeded
        """
        import json

        access = cls.get_consumer_app_access(consumer.id, app_id)

        if not access:
            raise ExternalConsumerAccessDeniedError(
                "Consumer does not have access to this app."
            )

        # Check validity period
        if not access.is_valid:
            raise ExternalConsumerAccessDeniedError("Access has expired or is not yet valid.")

        # Check invoke permission
        if not access.can_invoke:
            raise ExternalConsumerAccessDeniedError(
                "Consumer does not have invoke permission for this app."
            )

        # Check consumer-level quota
        if not consumer.is_within_quota:
            raise ExternalConsumerQuotaExceededError("Consumer quota exceeded.")

        # Check app-level quota
        if not access.is_within_quota:
            raise ExternalConsumerQuotaExceededError("App quota exceeded.")

        # Check scope if required
        if required_scope and access.allowed_scopes:
            allowed = json.loads(access.allowed_scopes)
            if required_scope not in allowed:
                raise ExternalConsumerAccessDeniedError(
                    f"Required scope '{required_scope}' not granted."
                )

        return access

    # ==========================================
    # Usage Tracking
    # ==========================================

    @classmethod
    def log_usage(
        cls,
        consumer_id: str,
        app_id: str,
        tenant_id: str,
        endpoint: str,
        method: str,
        status_code: int,
        response_time_ms: int,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        estimated_cost: float = 0.0,
        request_id: str | None = None,
        ip_address: str | None = None,
        user_agent: str | None = None,
    ) -> ExternalConsumerUsageLog:
        """Log API usage for a consumer."""
        log = ExternalConsumerUsageLog(
            consumer_id=consumer_id,
            app_id=app_id,
            tenant_id=tenant_id,
            endpoint=endpoint,
            method=method,
            status_code=status_code,
            response_time_ms=response_time_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            estimated_cost=estimated_cost,
            request_id=request_id,
            ip_address=ip_address,
            user_agent=user_agent,
        )
        db.session.add(log)

        # Update quota usage
        consumer = cls.get_consumer(consumer_id)
        if consumer:
            consumer.quota_used += 1

        access = cls.get_consumer_app_access(consumer_id, app_id)
        if access:
            access.app_quota_used += 1

        db.session.commit()
        return log

    @classmethod
    def get_consumer_usage_logs(
        cls,
        consumer_id: str,
        app_id: str | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Sequence[ExternalConsumerUsageLog]:
        """Get usage logs for a consumer with optional filtering."""
        query = select(ExternalConsumerUsageLog).where(
            ExternalConsumerUsageLog.consumer_id == consumer_id
        )

        if app_id:
            query = query.where(ExternalConsumerUsageLog.app_id == app_id)
        if start_date:
            query = query.where(ExternalConsumerUsageLog.created_at >= start_date)
        if end_date:
            query = query.where(ExternalConsumerUsageLog.created_at <= end_date)

        query = query.order_by(ExternalConsumerUsageLog.created_at.desc())
        query = query.offset(offset).limit(limit)

        return db.session.scalars(query).all()

    @classmethod
    def get_consumer_usage_stats(
        cls,
        consumer_id: str,
        app_id: str | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> dict:
        """
        Get aggregated usage statistics for a consumer.

        Returns:
            Dict with total_requests, total_tokens, total_cost, avg_response_time
        """
        from sqlalchemy import func as sa_func

        query = select(
            sa_func.count(ExternalConsumerUsageLog.id).label("total_requests"),
            sa_func.sum(ExternalConsumerUsageLog.total_tokens).label("total_tokens"),
            sa_func.sum(ExternalConsumerUsageLog.estimated_cost).label("total_cost"),
            sa_func.avg(ExternalConsumerUsageLog.response_time_ms).label("avg_response_time"),
        ).where(ExternalConsumerUsageLog.consumer_id == consumer_id)

        if app_id:
            query = query.where(ExternalConsumerUsageLog.app_id == app_id)
        if start_date:
            query = query.where(ExternalConsumerUsageLog.created_at >= start_date)
        if end_date:
            query = query.where(ExternalConsumerUsageLog.created_at <= end_date)

        result = db.session.execute(query).first()

        return {
            "total_requests": result.total_requests or 0,
            "total_tokens": result.total_tokens or 0,
            "total_cost": float(result.total_cost or 0),
            "avg_response_time": float(result.avg_response_time or 0),
        }

    @classmethod
    def reset_consumer_quota(cls, consumer_id: str) -> ExternalConsumer:
        """Reset a consumer's quota usage."""
        consumer = cls.get_consumer(consumer_id)
        if not consumer:
            raise ExternalConsumerNotFoundError()

        consumer.quota_used = 0
        consumer.quota_reset_at = datetime.utcnow()
        db.session.commit()
        return consumer

    @classmethod
    def reset_app_quota(cls, consumer_id: str, app_id: str) -> ExternalConsumerAppAccess:
        """Reset a consumer's quota usage for a specific app."""
        access = cls.get_consumer_app_access(consumer_id, app_id)
        if not access:
            raise ExternalConsumerAccessDeniedError("Access record not found.")

        access.app_quota_used = 0
        db.session.commit()
        return access
