"""
Unit tests for authentication system.
Tests JWT tokens, API keys, and auth middleware.
"""
import pytest
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, patch, AsyncMock

from app.core.auth import (
    AuthConfig,
    JWTHandler,
    APIKeyAuth,
    AuthMiddleware,
    User,
    TokenPayload,
    InvalidTokenError,
    ExpiredTokenError,
    InvalidAPIKeyError,
    hash_api_key,
    verify_api_key,
    generate_api_key,
)


class TestAuthConfig:
    """Tests for authentication configuration."""

    def test_default_config(self):
        """Test default auth configuration."""
        config = AuthConfig()
        assert config.jwt_secret is not None
        assert config.jwt_algorithm == "HS256"
        assert config.access_token_expire_minutes == 30
        assert config.refresh_token_expire_days == 7

    def test_config_from_env(self):
        """Test configuration from environment variables."""
        with patch.dict(
            "os.environ",
            {
                "JWT_SECRET": "test-secret-key",
                "JWT_ALGORITHM": "HS512",
                "ACCESS_TOKEN_EXPIRE_MINUTES": "60",
            },
        ):
            config = AuthConfig.from_env()
            assert config.jwt_secret == "test-secret-key"
            assert config.jwt_algorithm == "HS512"
            assert config.access_token_expire_minutes == 60


class TestJWTHandler:
    """Tests for JWT token handling."""

    @pytest.fixture
    def jwt_handler(self):
        """Create JWT handler with test config."""
        config = AuthConfig(jwt_secret="test-secret-key-for-testing")
        return JWTHandler(config)

    def test_create_access_token(self, jwt_handler):
        """Test creating an access token."""
        user = User(id="user-123", email="test@example.com")
        token = jwt_handler.create_access_token(user)

        assert token is not None
        assert isinstance(token, str)
        assert len(token) > 0

    def test_create_refresh_token(self, jwt_handler):
        """Test creating a refresh token."""
        user = User(id="user-123", email="test@example.com")
        token = jwt_handler.create_refresh_token(user)

        assert token is not None
        assert isinstance(token, str)

    def test_decode_valid_token(self, jwt_handler):
        """Test decoding a valid token."""
        user = User(id="user-123", email="test@example.com", roles=["admin"])
        token = jwt_handler.create_access_token(user)

        payload = jwt_handler.decode_token(token)

        assert payload.sub == "user-123"
        assert payload.email == "test@example.com"
        assert "admin" in payload.roles

    def test_decode_expired_token(self, jwt_handler):
        """Test decoding an expired token raises error."""
        user = User(id="user-123", email="test@example.com")
        # Create token that expires immediately
        token = jwt_handler.create_access_token(user, expires_delta=timedelta(seconds=-1))

        with pytest.raises(ExpiredTokenError):
            jwt_handler.decode_token(token)

    def test_decode_invalid_token(self, jwt_handler):
        """Test decoding an invalid token raises error."""
        with pytest.raises(InvalidTokenError):
            jwt_handler.decode_token("invalid.token.here")

    def test_decode_token_wrong_secret(self, jwt_handler):
        """Test decoding token with wrong secret raises error."""
        user = User(id="user-123", email="test@example.com")
        token = jwt_handler.create_access_token(user)

        # Create handler with different secret
        other_config = AuthConfig(jwt_secret="different-secret")
        other_handler = JWTHandler(other_config)

        with pytest.raises(InvalidTokenError):
            other_handler.decode_token(token)

    def test_refresh_token_has_longer_expiry(self, jwt_handler):
        """Test refresh token expires later than access token."""
        user = User(id="user-123", email="test@example.com")

        access_payload = jwt_handler.decode_token(
            jwt_handler.create_access_token(user)
        )
        refresh_payload = jwt_handler.decode_token(
            jwt_handler.create_refresh_token(user)
        )

        assert refresh_payload.exp > access_payload.exp


class TestAPIKeyAuth:
    """Tests for API key authentication."""

    def test_generate_api_key(self):
        """Test generating an API key."""
        key = generate_api_key()

        assert key is not None
        assert key.startswith("la_")  # LiteAgent prefix
        assert len(key) > 20

    def test_hash_api_key(self):
        """Test hashing an API key."""
        key = "la_test_key_12345"
        hashed = hash_api_key(key)

        assert hashed != key
        assert len(hashed) == 64  # SHA256 hex

    def test_verify_api_key(self):
        """Test verifying an API key."""
        key = "la_test_key_12345"
        hashed = hash_api_key(key)

        assert verify_api_key(key, hashed) is True
        assert verify_api_key("wrong_key", hashed) is False

    def test_api_key_auth_validate(self):
        """Test API key validation."""
        auth = APIKeyAuth()
        key = generate_api_key()
        hashed = hash_api_key(key)

        # Mock database lookup
        mock_get_key = AsyncMock(return_value={
            "id": "key-123",
            "user_id": "user-456",
            "hashed_key": hashed,
            "name": "Test Key",
            "is_active": True,
        })

        # Should validate successfully
        assert auth.verify(key, hashed) is True

    def test_api_key_auth_invalid_key(self):
        """Test invalid API key raises error."""
        auth = APIKeyAuth()
        hashed = hash_api_key("la_correct_key")

        with pytest.raises(InvalidAPIKeyError):
            auth.validate("la_wrong_key", hashed)


class TestUser:
    """Tests for User model."""

    def test_create_user(self):
        """Test creating a user."""
        user = User(
            id="user-123",
            email="test@example.com",
            roles=["user", "admin"],
        )

        assert user.id == "user-123"
        assert user.email == "test@example.com"
        assert "admin" in user.roles

    def test_user_has_role(self):
        """Test checking user roles."""
        user = User(id="user-123", email="test@example.com", roles=["admin"])

        assert user.has_role("admin") is True
        assert user.has_role("superuser") is False

    def test_user_default_roles(self):
        """Test user has default roles."""
        user = User(id="user-123", email="test@example.com")

        assert "user" in user.roles


class TestTokenPayload:
    """Tests for token payload."""

    def test_create_payload(self):
        """Test creating token payload."""
        payload = TokenPayload(
            sub="user-123",
            email="test@example.com",
            roles=["admin"],
            exp=datetime.now(timezone.utc) + timedelta(hours=1),
        )

        assert payload.sub == "user-123"
        assert payload.is_expired is False

    def test_payload_expired(self):
        """Test expired payload detection."""
        payload = TokenPayload(
            sub="user-123",
            email="test@example.com",
            exp=datetime.now(timezone.utc) - timedelta(hours=1),
        )

        assert payload.is_expired is True


class TestAuthMiddleware:
    """Tests for authentication middleware."""

    @pytest.fixture
    def middleware(self):
        """Create auth middleware."""
        config = AuthConfig(jwt_secret="test-secret")
        return AuthMiddleware(config)

    @pytest.mark.asyncio
    async def test_authenticate_bearer_token(self, middleware):
        """Test authenticating with bearer token."""
        jwt_handler = JWTHandler(AuthConfig(jwt_secret="test-secret"))
        user = User(id="user-123", email="test@example.com")
        token = jwt_handler.create_access_token(user)

        auth_user = await middleware.authenticate(f"Bearer {token}")

        assert auth_user.id == "user-123"

    @pytest.mark.asyncio
    async def test_authenticate_missing_token(self, middleware):
        """Test authentication fails without token."""
        with pytest.raises(InvalidTokenError):
            await middleware.authenticate(None)

    @pytest.mark.asyncio
    async def test_authenticate_invalid_scheme(self, middleware):
        """Test authentication fails with invalid scheme."""
        with pytest.raises(InvalidTokenError):
            await middleware.authenticate("Basic dXNlcjpwYXNz")

    @pytest.mark.asyncio
    async def test_skip_auth_for_public_paths(self, middleware):
        """Test public paths skip authentication."""
        result = middleware.should_skip_auth("/health")
        assert result is True

        result = middleware.should_skip_auth("/api/agents/")
        assert result is False
