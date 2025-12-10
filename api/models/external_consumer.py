"""
External Consumer Model - API consumers outside the workspace.

This module provides models for managing external API consumers,
their authentication, and per-consumer quotas/rate limits.
"""

import enum
import secrets
from datetime import datetime
from uuid import uuid4

import sqlalchemy as sa
from sqlalchemy import DateTime, String, func
from sqlalchemy.orm import Mapped, mapped_column

from .base import TypeBase
from .types import LongText, StringUUID


class ExternalConsumerStatus(enum.StrEnum):
    """Status of an external consumer."""
    ACTIVE = "active"
    SUSPENDED = "suspended"
    REVOKED = "revoked"


class ExternalConsumerAuthType(enum.StrEnum):
    """Authentication type for external consumers."""
    API_KEY = "api_key"
    OAUTH = "oauth"
    JWT = "jwt"


class ExternalConsumer(TypeBase):
    """
    External API consumers - entities that consume published apps.

    External consumers are separate from workspace members and can
    be individuals, organizations, or applications that access
    published apps via API.
    """
    __tablename__ = "external_consumers"
    __table_args__ = (
        sa.PrimaryKeyConstraint("id", name="external_consumer_pkey"),
        sa.Index("idx_external_consumers_tenant_id", "tenant_id"),
        sa.Index("idx_external_consumers_email", "email"),
        sa.Index("idx_external_consumers_status", "status"),
        sa.UniqueConstraint("tenant_id", "email", name="unique_external_consumer_email"),
    )

    id: Mapped[str] = mapped_column(
        StringUUID,
        insert_default=lambda: str(uuid4()),
        default_factory=lambda: str(uuid4()),
        primary_key=True,
        init=False,
    )
    tenant_id: Mapped[str] = mapped_column(StringUUID, nullable=False)

    # Consumer identity
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    email: Mapped[str] = mapped_column(String(255), nullable=False)
    organization: Mapped[str | None] = mapped_column(String(255), nullable=True, default=None)
    description: Mapped[str | None] = mapped_column(LongText, nullable=True, default=None)

    # Authentication
    auth_type: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        server_default=sa.text("'api_key'"),
        default=ExternalConsumerAuthType.API_KEY,
    )
    api_key_hash: Mapped[str | None] = mapped_column(String(255), nullable=True, default=None)
    api_key_prefix: Mapped[str | None] = mapped_column(String(16), nullable=True, default=None)

    # Status and lifecycle
    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        server_default=sa.text("'active'"),
        default=ExternalConsumerStatus.ACTIVE,
    )

    # Rate limiting (consumer-level limits)
    rate_limit_rpm: Mapped[int | None] = mapped_column(sa.Integer, nullable=True, default=None)
    rate_limit_rph: Mapped[int | None] = mapped_column(sa.Integer, nullable=True, default=None)
    rate_limit_rpd: Mapped[int | None] = mapped_column(sa.Integer, nullable=True, default=None)

    # Quota management
    quota_total: Mapped[int | None] = mapped_column(sa.Integer, nullable=True, default=None)
    quota_used: Mapped[int] = mapped_column(
        sa.Integer, nullable=False, server_default=sa.text("0"), default=0
    )
    quota_reset_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True, default=None)

    # Metadata
    metadata_json: Mapped[str | None] = mapped_column(LongText, nullable=True, default=None)
    webhook_url: Mapped[str | None] = mapped_column(String(512), nullable=True, default=None)

    # Audit
    created_by: Mapped[str] = mapped_column(StringUUID, nullable=False)
    last_active_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True, default=None)
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
        return f"<ExternalConsumer(id={self.id}, name={self.name}, email={self.email})>"

    @property
    def status_enum(self) -> ExternalConsumerStatus:
        """Get status as enum."""
        return ExternalConsumerStatus(self.status)

    @property
    def auth_type_enum(self) -> ExternalConsumerAuthType:
        """Get auth type as enum."""
        return ExternalConsumerAuthType(self.auth_type)

    @property
    def is_active(self) -> bool:
        """Check if consumer is active."""
        return self.status == ExternalConsumerStatus.ACTIVE

    @property
    def is_within_quota(self) -> bool:
        """Check if consumer is within their quota."""
        if self.quota_total is None:
            return True  # No quota limit
        return self.quota_used < self.quota_total

    @staticmethod
    def generate_api_key() -> tuple[str, str, str]:
        """
        Generate a new API key for the consumer.

        Returns:
            Tuple of (full_key, key_prefix, key_hash)
        """
        import hashlib

        # Generate random key
        key = f"ec_{secrets.token_urlsafe(32)}"
        prefix = key[:12]

        # Hash for storage
        key_hash = hashlib.sha256(key.encode()).hexdigest()

        return key, prefix, key_hash

    @staticmethod
    def verify_api_key(provided_key: str, stored_hash: str) -> bool:
        """Verify an API key against stored hash."""
        import hashlib
        return hashlib.sha256(provided_key.encode()).hexdigest() == stored_hash


class ExternalConsumerAppAccess(TypeBase):
    """
    Per-app access grants for external consumers.

    Links external consumers to specific published apps with
    optional custom rate limits and scopes.
    """
    __tablename__ = "external_consumer_app_access"
    __table_args__ = (
        sa.PrimaryKeyConstraint("id", name="external_consumer_app_access_pkey"),
        sa.Index("idx_external_consumer_app_access_consumer_id", "consumer_id"),
        sa.Index("idx_external_consumer_app_access_app_id", "app_id"),
        sa.UniqueConstraint("consumer_id", "app_id", name="unique_consumer_app_access"),
    )

    id: Mapped[str] = mapped_column(
        StringUUID,
        insert_default=lambda: str(uuid4()),
        default_factory=lambda: str(uuid4()),
        primary_key=True,
        init=False,
    )
    consumer_id: Mapped[str] = mapped_column(StringUUID, nullable=False)
    app_id: Mapped[str] = mapped_column(StringUUID, nullable=False)
    tenant_id: Mapped[str] = mapped_column(StringUUID, nullable=False)

    # Access level
    can_invoke: Mapped[bool] = mapped_column(
        sa.Boolean, nullable=False, server_default=sa.text("true"), default=True
    )
    can_view_logs: Mapped[bool] = mapped_column(
        sa.Boolean, nullable=False, server_default=sa.text("false"), default=False
    )

    # Custom rate limits (overrides consumer-level)
    custom_rate_limit_rpm: Mapped[int | None] = mapped_column(sa.Integer, nullable=True, default=None)
    custom_rate_limit_rph: Mapped[int | None] = mapped_column(sa.Integer, nullable=True, default=None)

    # Custom quota (per-app)
    app_quota_total: Mapped[int | None] = mapped_column(sa.Integer, nullable=True, default=None)
    app_quota_used: Mapped[int] = mapped_column(
        sa.Integer, nullable=False, server_default=sa.text("0"), default=0
    )

    # Allowed API scopes (JSON array of scope names)
    allowed_scopes: Mapped[str | None] = mapped_column(LongText, nullable=True, default=None)

    # Validity period
    valid_from: Mapped[datetime | None] = mapped_column(DateTime, nullable=True, default=None)
    valid_until: Mapped[datetime | None] = mapped_column(DateTime, nullable=True, default=None)

    # Audit
    granted_by: Mapped[str] = mapped_column(StringUUID, nullable=False)
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
        return f"<ExternalConsumerAppAccess(consumer_id={self.consumer_id}, app_id={self.app_id})>"

    @property
    def is_valid(self) -> bool:
        """Check if access is currently valid."""
        now = datetime.utcnow()
        if self.valid_from and now < self.valid_from:
            return False
        if self.valid_until and now > self.valid_until:
            return False
        return True

    @property
    def is_within_quota(self) -> bool:
        """Check if access is within app-specific quota."""
        if self.app_quota_total is None:
            return True
        return self.app_quota_used < self.app_quota_total


class ExternalConsumerUsageLog(TypeBase):
    """
    Usage logs for external consumers.

    Tracks API calls, token usage, and costs per consumer/app.
    """
    __tablename__ = "external_consumer_usage_logs"
    __table_args__ = (
        sa.PrimaryKeyConstraint("id", name="external_consumer_usage_log_pkey"),
        sa.Index("idx_external_consumer_usage_logs_consumer_id", "consumer_id"),
        sa.Index("idx_external_consumer_usage_logs_app_id", "app_id"),
        sa.Index("idx_external_consumer_usage_logs_created_at", "created_at"),
    )

    id: Mapped[str] = mapped_column(
        StringUUID,
        insert_default=lambda: str(uuid4()),
        default_factory=lambda: str(uuid4()),
        primary_key=True,
        init=False,
    )
    consumer_id: Mapped[str] = mapped_column(StringUUID, nullable=False)
    app_id: Mapped[str] = mapped_column(StringUUID, nullable=False)
    tenant_id: Mapped[str] = mapped_column(StringUUID, nullable=False)

    # Request details
    endpoint: Mapped[str] = mapped_column(String(255), nullable=False)
    method: Mapped[str] = mapped_column(String(16), nullable=False)
    status_code: Mapped[int] = mapped_column(sa.Integer, nullable=False)
    response_time_ms: Mapped[int] = mapped_column(sa.Integer, nullable=False)

    # Token usage
    prompt_tokens: Mapped[int] = mapped_column(
        sa.Integer, nullable=False, server_default=sa.text("0"), default=0
    )
    completion_tokens: Mapped[int] = mapped_column(
        sa.Integer, nullable=False, server_default=sa.text("0"), default=0
    )
    total_tokens: Mapped[int] = mapped_column(
        sa.Integer, nullable=False, server_default=sa.text("0"), default=0
    )

    # Cost tracking
    estimated_cost: Mapped[float] = mapped_column(
        sa.Float, nullable=False, server_default=sa.text("0"), default=0.0
    )

    # Request metadata
    request_id: Mapped[str | None] = mapped_column(String(64), nullable=True, default=None)
    ip_address: Mapped[str | None] = mapped_column(String(64), nullable=True, default=None)
    user_agent: Mapped[str | None] = mapped_column(String(512), nullable=True, default=None)

    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.current_timestamp(), init=False
    )

    def __repr__(self) -> str:
        return f"<ExternalConsumerUsageLog(consumer_id={self.consumer_id}, app_id={self.app_id})>"
