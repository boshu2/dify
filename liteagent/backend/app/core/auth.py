"""
Authentication system for LiteAgent API.
Supports JWT tokens and API key authentication.
"""
import os
import secrets
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import jwt


class InvalidTokenError(Exception):
    """Raised when token is invalid."""
    pass


class ExpiredTokenError(Exception):
    """Raised when token has expired."""
    pass


class InvalidAPIKeyError(Exception):
    """Raised when API key is invalid."""
    pass


@dataclass
class AuthConfig:
    """Authentication configuration."""

    jwt_secret: str = field(default_factory=lambda: secrets.token_hex(32))
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7
    api_key_prefix: str = "la_"

    @classmethod
    def from_env(cls) -> "AuthConfig":
        """Create configuration from environment variables."""
        return cls(
            jwt_secret=os.environ.get("JWT_SECRET", secrets.token_hex(32)),
            jwt_algorithm=os.environ.get("JWT_ALGORITHM", "HS256"),
            access_token_expire_minutes=int(
                os.environ.get("ACCESS_TOKEN_EXPIRE_MINUTES", "30")
            ),
            refresh_token_expire_days=int(
                os.environ.get("REFRESH_TOKEN_EXPIRE_DAYS", "7")
            ),
        )


@dataclass
class User:
    """User model for authentication."""

    id: str
    email: str
    roles: list[str] = field(default_factory=lambda: ["user"])
    metadata: dict[str, Any] = field(default_factory=dict)

    def has_role(self, role: str) -> bool:
        """Check if user has a specific role."""
        return role in self.roles


@dataclass
class TokenPayload:
    """JWT token payload."""

    sub: str  # User ID
    email: str
    exp: datetime
    roles: list[str] = field(default_factory=list)
    type: str = "access"  # access or refresh
    iat: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def is_expired(self) -> bool:
        """Check if token has expired."""
        return datetime.now(timezone.utc) > self.exp


class JWTHandler:
    """Handles JWT token creation and validation."""

    def __init__(self, config: AuthConfig):
        self.config = config

    def create_access_token(
        self,
        user: User,
        expires_delta: timedelta | None = None,
    ) -> str:
        """Create an access token for a user."""
        if expires_delta is None:
            expires_delta = timedelta(minutes=self.config.access_token_expire_minutes)

        return self._create_token(user, expires_delta, token_type="access")

    def create_refresh_token(
        self,
        user: User,
        expires_delta: timedelta | None = None,
    ) -> str:
        """Create a refresh token for a user."""
        if expires_delta is None:
            expires_delta = timedelta(days=self.config.refresh_token_expire_days)

        return self._create_token(user, expires_delta, token_type="refresh")

    def _create_token(
        self,
        user: User,
        expires_delta: timedelta,
        token_type: str,
    ) -> str:
        """Create a JWT token."""
        now = datetime.now(timezone.utc)
        expire = now + expires_delta

        payload = {
            "sub": user.id,
            "email": user.email,
            "roles": user.roles,
            "type": token_type,
            "iat": now.timestamp(),
            "exp": expire.timestamp(),
        }

        return jwt.encode(
            payload,
            self.config.jwt_secret,
            algorithm=self.config.jwt_algorithm,
        )

    def decode_token(self, token: str) -> TokenPayload:
        """Decode and validate a JWT token."""
        try:
            payload = jwt.decode(
                token,
                self.config.jwt_secret,
                algorithms=[self.config.jwt_algorithm],
            )

            return TokenPayload(
                sub=payload["sub"],
                email=payload["email"],
                roles=payload.get("roles", []),
                type=payload.get("type", "access"),
                exp=datetime.fromtimestamp(payload["exp"], tz=timezone.utc),
                iat=datetime.fromtimestamp(payload["iat"], tz=timezone.utc),
            )

        except jwt.ExpiredSignatureError:
            raise ExpiredTokenError("Token has expired")
        except jwt.InvalidTokenError as e:
            raise InvalidTokenError(f"Invalid token: {e}")


def generate_api_key(prefix: str = "la_") -> str:
    """Generate a new API key."""
    random_part = secrets.token_urlsafe(32)
    return f"{prefix}{random_part}"


def hash_api_key(key: str) -> str:
    """Hash an API key for storage."""
    return hashlib.sha256(key.encode()).hexdigest()


def verify_api_key(key: str, hashed: str) -> bool:
    """Verify an API key against its hash."""
    return secrets.compare_digest(hash_api_key(key), hashed)


class APIKeyAuth:
    """API key authentication handler."""

    def verify(self, key: str, stored_hash: str) -> bool:
        """Verify an API key against stored hash."""
        return verify_api_key(key, stored_hash)

    def validate(self, key: str, stored_hash: str) -> None:
        """Validate an API key, raising error if invalid."""
        if not self.verify(key, stored_hash):
            raise InvalidAPIKeyError("Invalid API key")


class AuthMiddleware:
    """Authentication middleware for FastAPI."""

    PUBLIC_PATHS = {
        "/",
        "/health",
        "/docs",
        "/openapi.json",
        "/redoc",
    }

    def __init__(self, config: AuthConfig):
        self.config = config
        self.jwt_handler = JWTHandler(config)

    def should_skip_auth(self, path: str) -> bool:
        """Check if path should skip authentication."""
        return path in self.PUBLIC_PATHS

    async def authenticate(self, authorization: str | None) -> User:
        """Authenticate a request from authorization header."""
        if not authorization:
            raise InvalidTokenError("Authorization header required")

        parts = authorization.split(" ")
        if len(parts) != 2:
            raise InvalidTokenError("Invalid authorization header format")

        scheme, token = parts

        if scheme.lower() == "bearer":
            return await self._authenticate_jwt(token)
        elif scheme.lower() == "apikey":
            return await self._authenticate_api_key(token)
        else:
            raise InvalidTokenError(f"Unsupported auth scheme: {scheme}")

    async def _authenticate_jwt(self, token: str) -> User:
        """Authenticate using JWT token."""
        payload = self.jwt_handler.decode_token(token)

        return User(
            id=payload.sub,
            email=payload.email,
            roles=payload.roles,
        )

    async def _authenticate_api_key(self, key: str) -> User:
        """Authenticate using API key."""
        # In production, this would look up the API key in the database
        # For now, just validate format
        if not key.startswith("la_"):
            raise InvalidAPIKeyError("Invalid API key format")

        # Return a service user for API key auth
        return User(
            id="api-key-user",
            email="api@liteagent.local",
            roles=["api"],
        )
