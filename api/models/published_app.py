"""
Published App Model - Apps exposed for external consumption.

This module provides models for managing published apps,
which are apps made available outside the workspace for
external consumers to access.
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


class PublishedAppStatus(enum.StrEnum):
    """Status of a published app."""
    DRAFT = "draft"        # Not yet published
    PUBLISHED = "published"  # Live and accessible
    PAUSED = "paused"      # Temporarily disabled
    DEPRECATED = "deprecated"  # Marked for removal
    ARCHIVED = "archived"  # Permanently disabled


class PublishedAppVisibility(enum.StrEnum):
    """Visibility level of a published app."""
    PRIVATE = "private"    # Only accessible with explicit grant
    UNLISTED = "unlisted"  # Accessible with link, not discoverable
    PUBLIC = "public"      # Publicly discoverable and accessible


class PublishedApp(TypeBase):
    """
    Published apps for external consumption.

    A published app is a wrapper around an internal app that
    exposes it for external consumers with its own configuration,
    rate limits, and access controls.
    """
    __tablename__ = "published_apps"
    __table_args__ = (
        sa.PrimaryKeyConstraint("id", name="published_app_pkey"),
        sa.Index("idx_published_apps_tenant_id", "tenant_id"),
        sa.Index("idx_published_apps_app_id", "app_id"),
        sa.Index("idx_published_apps_slug", "slug"),
        sa.Index("idx_published_apps_status", "status"),
        sa.UniqueConstraint("slug", name="unique_published_app_slug"),
        sa.UniqueConstraint("tenant_id", "app_id", name="unique_published_app_per_tenant_app"),
    )

    id: Mapped[str] = mapped_column(
        StringUUID,
        insert_default=lambda: str(uuid4()),
        default_factory=lambda: str(uuid4()),
        primary_key=True,
        init=False,
    )
    tenant_id: Mapped[str] = mapped_column(StringUUID, nullable=False)
    app_id: Mapped[str] = mapped_column(StringUUID, nullable=False)

    # Public identity
    slug: Mapped[str] = mapped_column(String(128), nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(LongText, nullable=True, default=None)
    icon: Mapped[str | None] = mapped_column(String(255), nullable=True, default=None)
    icon_background: Mapped[str | None] = mapped_column(String(32), nullable=True, default=None)

    # Versioning
    version: Mapped[str] = mapped_column(String(32), nullable=False, default="1.0.0")
    changelog: Mapped[str | None] = mapped_column(LongText, nullable=True, default=None)

    # Status and visibility
    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        server_default=sa.text("'draft'"),
        default=PublishedAppStatus.DRAFT,
    )
    visibility: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        server_default=sa.text("'private'"),
        default=PublishedAppVisibility.PRIVATE,
    )

    # Default rate limits for consumers
    default_rate_limit_rpm: Mapped[int] = mapped_column(
        sa.Integer, nullable=False, server_default=sa.text("60"), default=60
    )
    default_rate_limit_rph: Mapped[int] = mapped_column(
        sa.Integer, nullable=False, server_default=sa.text("1000"), default=1000
    )
    default_rate_limit_rpd: Mapped[int | None] = mapped_column(sa.Integer, nullable=True, default=None)

    # Pricing/quota configuration
    free_quota_per_consumer: Mapped[int | None] = mapped_column(sa.Integer, nullable=True, default=None)
    pricing_model: Mapped[str | None] = mapped_column(String(32), nullable=True, default=None)
    price_per_request: Mapped[float | None] = mapped_column(sa.Float, nullable=True, default=None)

    # Terms and documentation
    terms_of_service: Mapped[str | None] = mapped_column(LongText, nullable=True, default=None)
    privacy_policy: Mapped[str | None] = mapped_column(LongText, nullable=True, default=None)
    documentation_url: Mapped[str | None] = mapped_column(String(512), nullable=True, default=None)
    support_email: Mapped[str | None] = mapped_column(String(255), nullable=True, default=None)

    # Custom configuration
    custom_domain: Mapped[str | None] = mapped_column(String(255), nullable=True, default=None)
    cors_allowed_origins: Mapped[str | None] = mapped_column(LongText, nullable=True, default=None)
    webhook_url: Mapped[str | None] = mapped_column(String(512), nullable=True, default=None)

    # Analytics
    total_consumers: Mapped[int] = mapped_column(
        sa.Integer, nullable=False, server_default=sa.text("0"), default=0
    )
    total_requests: Mapped[int] = mapped_column(
        sa.Integer, nullable=False, server_default=sa.text("0"), default=0
    )

    # Audit
    published_by: Mapped[str | None] = mapped_column(StringUUID, nullable=True, default=None)
    published_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True, default=None)
    created_by: Mapped[str] = mapped_column(StringUUID, nullable=False)
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
        return f"<PublishedApp(id={self.id}, slug={self.slug}, status={self.status})>"

    @property
    def status_enum(self) -> PublishedAppStatus:
        """Get status as enum."""
        return PublishedAppStatus(self.status)

    @property
    def visibility_enum(self) -> PublishedAppVisibility:
        """Get visibility as enum."""
        return PublishedAppVisibility(self.visibility)

    @property
    def is_live(self) -> bool:
        """Check if the app is currently live and accessible."""
        return self.status == PublishedAppStatus.PUBLISHED

    @property
    def is_public(self) -> bool:
        """Check if the app is publicly accessible."""
        return self.visibility == PublishedAppVisibility.PUBLIC and self.is_live

    @property
    def api_base_url(self) -> str:
        """Get the base URL for API access."""
        from configs import dify_config
        base = dify_config.SERVICE_API_URL or ""
        return f"{base.rstrip('/')}/published/{self.slug}"

    @staticmethod
    def generate_slug(name: str) -> str:
        """Generate a unique slug from a name."""
        import re
        # Convert to lowercase, replace spaces with hyphens
        slug = name.lower().strip()
        slug = re.sub(r'[^\w\s-]', '', slug)
        slug = re.sub(r'[\s_]+', '-', slug)
        # Add random suffix for uniqueness
        suffix = secrets.token_hex(4)
        return f"{slug[:100]}-{suffix}"


class PublishedAppVersion(TypeBase):
    """
    Version history for published apps.

    Tracks changes to published apps over time, allowing
    rollback and version-specific access.
    """
    __tablename__ = "published_app_versions"
    __table_args__ = (
        sa.PrimaryKeyConstraint("id", name="published_app_version_pkey"),
        sa.Index("idx_published_app_versions_published_app_id", "published_app_id"),
        sa.Index("idx_published_app_versions_version", "version"),
        sa.UniqueConstraint("published_app_id", "version", name="unique_published_app_version"),
    )

    id: Mapped[str] = mapped_column(
        StringUUID,
        insert_default=lambda: str(uuid4()),
        default_factory=lambda: str(uuid4()),
        primary_key=True,
        init=False,
    )
    published_app_id: Mapped[str] = mapped_column(StringUUID, nullable=False)
    version: Mapped[str] = mapped_column(String(32), nullable=False)

    # Snapshot of configuration at this version
    workflow_version: Mapped[str | None] = mapped_column(String(255), nullable=True, default=None)
    config_snapshot: Mapped[str | None] = mapped_column(LongText, nullable=True, default=None)
    changelog: Mapped[str | None] = mapped_column(LongText, nullable=True, default=None)

    # Version status
    is_current: Mapped[bool] = mapped_column(
        sa.Boolean, nullable=False, server_default=sa.text("false"), default=False
    )
    is_deprecated: Mapped[bool] = mapped_column(
        sa.Boolean, nullable=False, server_default=sa.text("false"), default=False
    )
    deprecation_message: Mapped[str | None] = mapped_column(LongText, nullable=True, default=None)

    # Audit
    created_by: Mapped[str] = mapped_column(StringUUID, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.current_timestamp(), init=False
    )

    def __repr__(self) -> str:
        return f"<PublishedAppVersion(published_app_id={self.published_app_id}, version={self.version})>"


class PublishedAppWebhook(TypeBase):
    """
    Webhook configuration for published apps.

    Allows external systems to receive notifications about
    events related to published apps.
    """
    __tablename__ = "published_app_webhooks"
    __table_args__ = (
        sa.PrimaryKeyConstraint("id", name="published_app_webhook_pkey"),
        sa.Index("idx_published_app_webhooks_published_app_id", "published_app_id"),
    )

    id: Mapped[str] = mapped_column(
        StringUUID,
        insert_default=lambda: str(uuid4()),
        default_factory=lambda: str(uuid4()),
        primary_key=True,
        init=False,
    )
    published_app_id: Mapped[str] = mapped_column(StringUUID, nullable=False)
    tenant_id: Mapped[str] = mapped_column(StringUUID, nullable=False)

    # Webhook configuration
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    url: Mapped[str] = mapped_column(String(512), nullable=False)
    secret: Mapped[str | None] = mapped_column(String(255), nullable=True, default=None)

    # Events to trigger
    events: Mapped[str] = mapped_column(LongText, nullable=False)  # JSON array of event types

    # Status
    is_active: Mapped[bool] = mapped_column(
        sa.Boolean, nullable=False, server_default=sa.text("true"), default=True
    )

    # Failure tracking
    consecutive_failures: Mapped[int] = mapped_column(
        sa.Integer, nullable=False, server_default=sa.text("0"), default=0
    )
    last_failure_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True, default=None)
    last_failure_reason: Mapped[str | None] = mapped_column(LongText, nullable=True, default=None)

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
        return f"<PublishedAppWebhook(published_app_id={self.published_app_id}, url={self.url})>"
