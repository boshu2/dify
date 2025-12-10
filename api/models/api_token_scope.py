"""
API Token Scope Model - Scoped API keys for fine-grained access control.

This module extends the API token system with scoping capabilities,
allowing tokens to be restricted to specific endpoints and operations.
"""

import enum
from datetime import datetime
from uuid import uuid4

import sqlalchemy as sa
from sqlalchemy import DateTime, String, func
from sqlalchemy.orm import Mapped, mapped_column

from .base import TypeBase
from .types import LongText, StringUUID


class ApiScope(enum.StrEnum):
    """
    API scopes for granular permission control.

    Scopes follow a resource:action pattern.
    """
    # App invocation scopes
    APP_INVOKE = "app:invoke"
    APP_INVOKE_STREAM = "app:invoke:stream"

    # Conversation scopes
    CONVERSATION_READ = "conversation:read"
    CONVERSATION_WRITE = "conversation:write"
    CONVERSATION_DELETE = "conversation:delete"

    # Message scopes
    MESSAGE_READ = "message:read"
    MESSAGE_WRITE = "message:write"
    MESSAGE_FEEDBACK = "message:feedback"

    # File scopes
    FILE_UPLOAD = "file:upload"
    FILE_READ = "file:read"

    # Workflow scopes
    WORKFLOW_RUN = "workflow:run"
    WORKFLOW_STOP = "workflow:stop"
    WORKFLOW_READ = "workflow:read"

    # Audio scopes
    AUDIO_SPEECH_TO_TEXT = "audio:speech_to_text"
    AUDIO_TEXT_TO_SPEECH = "audio:text_to_speech"

    # Meta scopes
    META_READ = "meta:read"
    PARAMETERS_READ = "parameters:read"

    # Admin scopes (for internal use)
    ADMIN_ALL = "admin:all"

    @classmethod
    def get_default_scopes(cls) -> list["ApiScope"]:
        """Get default scopes for new API tokens (backward compatible)."""
        return [
            cls.APP_INVOKE,
            cls.APP_INVOKE_STREAM,
            cls.CONVERSATION_READ,
            cls.CONVERSATION_WRITE,
            cls.MESSAGE_READ,
            cls.MESSAGE_WRITE,
            cls.MESSAGE_FEEDBACK,
            cls.FILE_UPLOAD,
            cls.FILE_READ,
            cls.WORKFLOW_RUN,
            cls.AUDIO_SPEECH_TO_TEXT,
            cls.AUDIO_TEXT_TO_SPEECH,
            cls.META_READ,
            cls.PARAMETERS_READ,
        ]

    @classmethod
    def get_read_only_scopes(cls) -> list["ApiScope"]:
        """Get read-only scopes for restricted tokens."""
        return [
            cls.CONVERSATION_READ,
            cls.MESSAGE_READ,
            cls.FILE_READ,
            cls.META_READ,
            cls.PARAMETERS_READ,
        ]

    @classmethod
    def get_invoke_scopes(cls) -> list["ApiScope"]:
        """Get scopes needed for app invocation only."""
        return [
            cls.APP_INVOKE,
            cls.APP_INVOKE_STREAM,
            cls.MESSAGE_READ,
            cls.META_READ,
            cls.PARAMETERS_READ,
        ]


class ApiTokenScope(TypeBase):
    """
    Scope configuration for API tokens.

    Links API tokens to their allowed scopes, providing
    fine-grained access control per token.
    """
    __tablename__ = "api_token_scopes"
    __table_args__ = (
        sa.PrimaryKeyConstraint("id", name="api_token_scope_pkey"),
        sa.Index("idx_api_token_scopes_token_id", "token_id"),
        sa.UniqueConstraint("token_id", "scope", name="unique_token_scope"),
    )

    id: Mapped[str] = mapped_column(
        StringUUID,
        insert_default=lambda: str(uuid4()),
        default_factory=lambda: str(uuid4()),
        primary_key=True,
        init=False,
    )
    token_id: Mapped[str] = mapped_column(StringUUID, nullable=False)
    scope: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.current_timestamp(), init=False
    )

    def __repr__(self) -> str:
        return f"<ApiTokenScope(token_id={self.token_id}, scope={self.scope})>"


class ApiTokenConfig(TypeBase):
    """
    Extended configuration for API tokens.

    Provides additional settings beyond basic token attributes,
    including expiration, rate limits, and usage tracking.
    """
    __tablename__ = "api_token_configs"
    __table_args__ = (
        sa.PrimaryKeyConstraint("id", name="api_token_config_pkey"),
        sa.Index("idx_api_token_configs_token_id", "token_id"),
        sa.UniqueConstraint("token_id", name="unique_api_token_config_token_id"),
    )

    id: Mapped[str] = mapped_column(
        StringUUID,
        insert_default=lambda: str(uuid4()),
        default_factory=lambda: str(uuid4()),
        primary_key=True,
        init=False,
    )
    token_id: Mapped[str] = mapped_column(StringUUID, nullable=False)

    # Token metadata
    name: Mapped[str | None] = mapped_column(String(255), nullable=True, default=None)
    description: Mapped[str | None] = mapped_column(LongText, nullable=True, default=None)

    # Expiration
    expires_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True, default=None)
    is_expired: Mapped[bool] = mapped_column(
        sa.Boolean, nullable=False, server_default=sa.text("false"), default=False
    )

    # Rate limiting (token-level)
    rate_limit_rpm: Mapped[int | None] = mapped_column(sa.Integer, nullable=True, default=None)
    rate_limit_rph: Mapped[int | None] = mapped_column(sa.Integer, nullable=True, default=None)

    # Usage tracking
    total_requests: Mapped[int] = mapped_column(
        sa.Integer, nullable=False, server_default=sa.text("0"), default=0
    )
    total_tokens_used: Mapped[int] = mapped_column(
        sa.Integer, nullable=False, server_default=sa.text("0"), default=0
    )
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True, default=None)
    last_used_ip: Mapped[str | None] = mapped_column(String(64), nullable=True, default=None)

    # IP restrictions (JSON array of allowed IPs/CIDR ranges)
    allowed_ips: Mapped[str | None] = mapped_column(LongText, nullable=True, default=None)

    # Referrer restrictions (JSON array of allowed referrer patterns)
    allowed_referrers: Mapped[str | None] = mapped_column(LongText, nullable=True, default=None)

    # External consumer link (if token belongs to external consumer)
    external_consumer_id: Mapped[str | None] = mapped_column(StringUUID, nullable=True, default=None)

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
        return f"<ApiTokenConfig(token_id={self.token_id}, name={self.name})>"

    @property
    def is_token_expired(self) -> bool:
        """Check if the token has expired."""
        if self.is_expired:
            return True
        if self.expires_at and datetime.utcnow() > self.expires_at:
            return True
        return False

    def check_ip_allowed(self, ip_address: str) -> bool:
        """Check if an IP address is allowed."""
        import ipaddress
        import json

        if not self.allowed_ips:
            return True  # No restrictions

        try:
            allowed = json.loads(self.allowed_ips)
            if not allowed:
                return True

            ip = ipaddress.ip_address(ip_address)
            for allowed_ip in allowed:
                try:
                    if "/" in allowed_ip:
                        # CIDR range
                        if ip in ipaddress.ip_network(allowed_ip, strict=False):
                            return True
                    else:
                        # Single IP
                        if ip == ipaddress.ip_address(allowed_ip):
                            return True
                except ValueError:
                    continue

            return False
        except (json.JSONDecodeError, ValueError):
            return True  # Invalid config = allow all

    def check_referrer_allowed(self, referrer: str | None) -> bool:
        """Check if a referrer is allowed."""
        import fnmatch
        import json

        if not self.allowed_referrers:
            return True  # No restrictions

        if not referrer:
            return False  # Referrer required but not provided

        try:
            allowed = json.loads(self.allowed_referrers)
            if not allowed:
                return True

            for pattern in allowed:
                if fnmatch.fnmatch(referrer, pattern):
                    return True

            return False
        except json.JSONDecodeError:
            return True  # Invalid config = allow all
