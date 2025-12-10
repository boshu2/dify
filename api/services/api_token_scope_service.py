"""
API Token Scope Service - Business logic for scoped API tokens.

This service provides methods for managing API token scopes,
checking scope permissions, and enforcing scope-based access control.
"""

from collections.abc import Sequence
from datetime import datetime

from sqlalchemy import select

from extensions.ext_database import db
from models.api_token_scope import (
    ApiScope,
    ApiTokenConfig,
    ApiTokenScope,
)
from models.model import ApiToken


class ScopeNotAllowedError(Exception):
    """Raised when an API token does not have the required scope."""

    def __init__(self, scope: str, message: str | None = None):
        self.scope = scope
        self.message = message or f"API token does not have the required scope: {scope}"
        super().__init__(self.message)


class TokenExpiredError(Exception):
    """Raised when an API token has expired."""

    def __init__(self, message: str = "API token has expired."):
        self.message = message
        super().__init__(self.message)


class TokenIPRestrictionError(Exception):
    """Raised when an API token is used from a disallowed IP."""

    def __init__(self, message: str = "API token is not allowed from this IP address."):
        self.message = message
        super().__init__(self.message)


class ApiTokenScopeService:
    """Service for managing API token scopes."""

    # ==========================================
    # Scope Management Methods
    # ==========================================

    @classmethod
    def get_token_scopes(cls, token_id: str) -> list[str]:
        """
        Get all scopes assigned to a token.

        Args:
            token_id: The API token ID

        Returns:
            List of scope strings
        """
        scopes = db.session.scalars(
            select(ApiTokenScope.scope).where(ApiTokenScope.token_id == token_id)
        ).all()
        return list(scopes)

    @classmethod
    def has_scope(cls, token_id: str, scope: str | ApiScope) -> bool:
        """
        Check if a token has a specific scope.

        Args:
            token_id: The API token ID
            scope: The scope to check

        Returns:
            True if token has the scope
        """
        scope_str = scope.value if isinstance(scope, ApiScope) else scope

        # Check for admin:all scope (grants everything)
        if cls.has_admin_scope(token_id):
            return True

        result = db.session.scalar(
            select(ApiTokenScope.id).where(
                ApiTokenScope.token_id == token_id,
                ApiTokenScope.scope == scope_str,
            )
        )
        return result is not None

    @classmethod
    def has_admin_scope(cls, token_id: str) -> bool:
        """Check if token has admin:all scope."""
        result = db.session.scalar(
            select(ApiTokenScope.id).where(
                ApiTokenScope.token_id == token_id,
                ApiTokenScope.scope == ApiScope.ADMIN_ALL.value,
            )
        )
        return result is not None

    @classmethod
    def has_any_scope(cls, token_id: str, scopes: list[str | ApiScope]) -> bool:
        """
        Check if a token has any of the given scopes.

        Args:
            token_id: The API token ID
            scopes: List of scopes to check

        Returns:
            True if token has at least one of the scopes
        """
        if cls.has_admin_scope(token_id):
            return True

        scope_strs = [s.value if isinstance(s, ApiScope) else s for s in scopes]

        result = db.session.scalar(
            select(ApiTokenScope.id).where(
                ApiTokenScope.token_id == token_id,
                ApiTokenScope.scope.in_(scope_strs),
            )
        )
        return result is not None

    @classmethod
    def has_all_scopes(cls, token_id: str, scopes: list[str | ApiScope]) -> bool:
        """
        Check if a token has all of the given scopes.

        Args:
            token_id: The API token ID
            scopes: List of scopes to check

        Returns:
            True if token has all of the scopes
        """
        if cls.has_admin_scope(token_id):
            return True

        for scope in scopes:
            if not cls.has_scope(token_id, scope):
                return False
        return True

    @classmethod
    def grant_scope(cls, token_id: str, scope: str | ApiScope) -> ApiTokenScope:
        """
        Grant a scope to a token.

        Args:
            token_id: The API token ID
            scope: The scope to grant

        Returns:
            The created ApiTokenScope
        """
        scope_str = scope.value if isinstance(scope, ApiScope) else scope

        # Check if already exists
        existing = db.session.scalar(
            select(ApiTokenScope).where(
                ApiTokenScope.token_id == token_id,
                ApiTokenScope.scope == scope_str,
            )
        )
        if existing:
            return existing

        token_scope = ApiTokenScope(
            token_id=token_id,
            scope=scope_str,
        )
        db.session.add(token_scope)
        db.session.commit()
        return token_scope

    @classmethod
    def grant_scopes(
        cls, token_id: str, scopes: list[str | ApiScope]
    ) -> list[ApiTokenScope]:
        """
        Grant multiple scopes to a token.

        Args:
            token_id: The API token ID
            scopes: List of scopes to grant

        Returns:
            List of created ApiTokenScopes
        """
        result = []
        for scope in scopes:
            token_scope = cls.grant_scope(token_id, scope)
            result.append(token_scope)
        return result

    @classmethod
    def grant_default_scopes(cls, token_id: str) -> list[ApiTokenScope]:
        """Grant default scopes to a token (backward compatible)."""
        return cls.grant_scopes(token_id, ApiScope.get_default_scopes())

    @classmethod
    def grant_read_only_scopes(cls, token_id: str) -> list[ApiTokenScope]:
        """Grant read-only scopes to a token."""
        return cls.grant_scopes(token_id, ApiScope.get_read_only_scopes())

    @classmethod
    def grant_invoke_scopes(cls, token_id: str) -> list[ApiTokenScope]:
        """Grant invoke-only scopes to a token."""
        return cls.grant_scopes(token_id, ApiScope.get_invoke_scopes())

    @classmethod
    def revoke_scope(cls, token_id: str, scope: str | ApiScope) -> bool:
        """
        Revoke a scope from a token.

        Args:
            token_id: The API token ID
            scope: The scope to revoke

        Returns:
            True if scope was revoked, False if not found
        """
        scope_str = scope.value if isinstance(scope, ApiScope) else scope

        token_scope = db.session.scalar(
            select(ApiTokenScope).where(
                ApiTokenScope.token_id == token_id,
                ApiTokenScope.scope == scope_str,
            )
        )
        if not token_scope:
            return False

        db.session.delete(token_scope)
        db.session.commit()
        return True

    @classmethod
    def revoke_all_scopes(cls, token_id: str) -> int:
        """
        Revoke all scopes from a token.

        Args:
            token_id: The API token ID

        Returns:
            Number of scopes revoked
        """
        result = db.session.execute(
            ApiTokenScope.__table__.delete().where(ApiTokenScope.token_id == token_id)
        )
        db.session.commit()
        return result.rowcount

    @classmethod
    def replace_scopes(
        cls, token_id: str, scopes: list[str | ApiScope]
    ) -> list[ApiTokenScope]:
        """
        Replace all scopes for a token with new ones.

        Args:
            token_id: The API token ID
            scopes: New list of scopes

        Returns:
            List of new ApiTokenScopes
        """
        cls.revoke_all_scopes(token_id)
        return cls.grant_scopes(token_id, scopes)

    # ==========================================
    # Token Config Methods
    # ==========================================

    @classmethod
    def get_token_config(cls, token_id: str) -> ApiTokenConfig | None:
        """Get the configuration for a token."""
        return db.session.scalar(
            select(ApiTokenConfig).where(ApiTokenConfig.token_id == token_id)
        )

    @classmethod
    def create_or_update_config(
        cls,
        token_id: str,
        name: str | None = None,
        description: str | None = None,
        expires_at: datetime | None = None,
        rate_limit_rpm: int | None = None,
        rate_limit_rph: int | None = None,
        allowed_ips: list[str] | None = None,
        allowed_referrers: list[str] | None = None,
        external_consumer_id: str | None = None,
    ) -> ApiTokenConfig:
        """
        Create or update token configuration.

        Args:
            token_id: The API token ID
            name: Human-readable name for the token
            description: Description of token purpose
            expires_at: When the token expires
            rate_limit_rpm: Requests per minute limit
            rate_limit_rph: Requests per hour limit
            allowed_ips: List of allowed IP addresses/CIDR ranges
            allowed_referrers: List of allowed referrer patterns
            external_consumer_id: ID of external consumer if applicable

        Returns:
            The created or updated ApiTokenConfig
        """
        import json

        existing = cls.get_token_config(token_id)

        allowed_ips_json = json.dumps(allowed_ips) if allowed_ips else None
        allowed_referrers_json = json.dumps(allowed_referrers) if allowed_referrers else None

        if existing:
            if name is not None:
                existing.name = name
            if description is not None:
                existing.description = description
            if expires_at is not None:
                existing.expires_at = expires_at
            if rate_limit_rpm is not None:
                existing.rate_limit_rpm = rate_limit_rpm
            if rate_limit_rph is not None:
                existing.rate_limit_rph = rate_limit_rph
            if allowed_ips is not None:
                existing.allowed_ips = allowed_ips_json
            if allowed_referrers is not None:
                existing.allowed_referrers = allowed_referrers_json
            if external_consumer_id is not None:
                existing.external_consumer_id = external_consumer_id
            db.session.commit()
            return existing

        config = ApiTokenConfig(
            token_id=token_id,
            name=name,
            description=description,
            expires_at=expires_at,
            rate_limit_rpm=rate_limit_rpm,
            rate_limit_rph=rate_limit_rph,
            allowed_ips=allowed_ips_json,
            allowed_referrers=allowed_referrers_json,
            external_consumer_id=external_consumer_id,
        )
        db.session.add(config)
        db.session.commit()
        return config

    @classmethod
    def delete_config(cls, token_id: str) -> bool:
        """Delete token configuration."""
        config = cls.get_token_config(token_id)
        if not config:
            return False

        db.session.delete(config)
        db.session.commit()
        return True

    # ==========================================
    # Token Validation Methods
    # ==========================================

    @classmethod
    def validate_token_scope(
        cls,
        token_id: str,
        required_scope: str | ApiScope,
    ) -> None:
        """
        Validate that a token has the required scope.

        Args:
            token_id: The API token ID
            required_scope: The scope required for the operation

        Raises:
            ScopeNotAllowedError: If token doesn't have the required scope
        """
        if not cls.has_scope(token_id, required_scope):
            scope_str = required_scope.value if isinstance(required_scope, ApiScope) else required_scope
            raise ScopeNotAllowedError(scope_str)

    @classmethod
    def validate_token_config(
        cls,
        token_id: str,
        ip_address: str | None = None,
        referrer: str | None = None,
    ) -> ApiTokenConfig | None:
        """
        Validate token configuration (expiration, IP, referrer).

        Args:
            token_id: The API token ID
            ip_address: The client IP address
            referrer: The request referrer

        Returns:
            The token config if valid

        Raises:
            TokenExpiredError: If token has expired
            TokenIPRestrictionError: If IP is not allowed
        """
        config = cls.get_token_config(token_id)

        if not config:
            return None  # No config = no restrictions

        # Check expiration
        if config.is_token_expired:
            raise TokenExpiredError()

        # Check IP restriction
        if ip_address and not config.check_ip_allowed(ip_address):
            raise TokenIPRestrictionError()

        # Check referrer restriction
        if config.allowed_referrers and not config.check_referrer_allowed(referrer):
            raise TokenIPRestrictionError("Referrer not allowed.")

        return config

    @classmethod
    def record_token_usage(
        cls,
        token_id: str,
        ip_address: str | None = None,
        tokens_used: int = 0,
    ) -> None:
        """
        Record token usage (for analytics and rate limiting).

        Args:
            token_id: The API token ID
            ip_address: The client IP address
            tokens_used: Number of tokens consumed (for LLM calls)
        """
        config = cls.get_token_config(token_id)

        if config:
            config.total_requests += 1
            config.total_tokens_used += tokens_used
            config.last_used_at = datetime.utcnow()
            config.last_used_ip = ip_address
            db.session.commit()

    # ==========================================
    # Token Lookup Methods
    # ==========================================

    @classmethod
    def get_tokens_by_external_consumer(
        cls, external_consumer_id: str
    ) -> Sequence[ApiTokenConfig]:
        """Get all token configs for an external consumer."""
        return db.session.scalars(
            select(ApiTokenConfig).where(
                ApiTokenConfig.external_consumer_id == external_consumer_id
            )
        ).all()

    @classmethod
    def get_tokens_by_scope(
        cls, scope: str | ApiScope
    ) -> Sequence[ApiTokenScope]:
        """Get all token-scope associations for a specific scope."""
        scope_str = scope.value if isinstance(scope, ApiScope) else scope
        return db.session.scalars(
            select(ApiTokenScope).where(ApiTokenScope.scope == scope_str)
        ).all()

    @classmethod
    def get_expiring_tokens(
        cls, within_days: int = 7
    ) -> Sequence[ApiTokenConfig]:
        """Get tokens expiring within a certain number of days."""
        cutoff = datetime.utcnow()
        from datetime import timedelta
        expiry_threshold = cutoff + timedelta(days=within_days)

        return db.session.scalars(
            select(ApiTokenConfig).where(
                ApiTokenConfig.expires_at.isnot(None),
                ApiTokenConfig.expires_at <= expiry_threshold,
                ApiTokenConfig.is_expired == False,  # noqa: E712
            )
        ).all()

    # ==========================================
    # Scope Information Methods
    # ==========================================

    @classmethod
    def get_all_scopes(cls) -> list[str]:
        """Get list of all available scopes."""
        return [scope.value for scope in ApiScope]

    @classmethod
    def get_scope_description(cls, scope: str | ApiScope) -> str:
        """Get human-readable description for a scope."""
        scope_str = scope.value if isinstance(scope, ApiScope) else scope

        descriptions = {
            ApiScope.APP_INVOKE.value: "Invoke the app (non-streaming)",
            ApiScope.APP_INVOKE_STREAM.value: "Invoke the app with streaming",
            ApiScope.CONVERSATION_READ.value: "Read conversation history",
            ApiScope.CONVERSATION_WRITE.value: "Create new conversations",
            ApiScope.CONVERSATION_DELETE.value: "Delete conversations",
            ApiScope.MESSAGE_READ.value: "Read messages",
            ApiScope.MESSAGE_WRITE.value: "Send messages",
            ApiScope.MESSAGE_FEEDBACK.value: "Submit message feedback",
            ApiScope.FILE_UPLOAD.value: "Upload files",
            ApiScope.FILE_READ.value: "Read/download files",
            ApiScope.WORKFLOW_RUN.value: "Run workflows",
            ApiScope.WORKFLOW_STOP.value: "Stop running workflows",
            ApiScope.WORKFLOW_READ.value: "Read workflow status",
            ApiScope.AUDIO_SPEECH_TO_TEXT.value: "Convert speech to text",
            ApiScope.AUDIO_TEXT_TO_SPEECH.value: "Convert text to speech",
            ApiScope.META_READ.value: "Read app metadata",
            ApiScope.PARAMETERS_READ.value: "Read app parameters",
            ApiScope.ADMIN_ALL.value: "Full administrative access",
        }

        return descriptions.get(scope_str, f"Unknown scope: {scope_str}")
