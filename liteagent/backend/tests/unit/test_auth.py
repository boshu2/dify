"""Tests for authentication system."""
import pytest
from unittest.mock import AsyncMock, MagicMock

from app.core.auth import (
    AuthConfig,
    JWTHandler,
    User,
    TokenPayload,
    InvalidTokenError,
    ExpiredTokenError,
    generate_api_key,
    hash_api_key,
    verify_api_key,
)
from app.services.auth_service import (
    AuthService,
    hash_password,
    verify_password,
    AuthenticationError,
    UserExistsError,
)


class TestPasswordHashing:
    """Tests for password hashing."""

    def test_hash_password(self):
        """Test password hashing."""
        password = "test_password_123"
        hashed = hash_password(password)

        # Should contain salt and hash separated by colon
        assert ":" in hashed
        salt, hash_value = hashed.split(":")
        assert len(salt) == 32  # 16 bytes hex = 32 chars
        assert len(hash_value) == 64  # SHA256 = 64 hex chars

    def test_verify_password_correct(self):
        """Test verifying correct password."""
        password = "my_secure_password"
        hashed = hash_password(password)

        assert verify_password(password, hashed) is True

    def test_verify_password_incorrect(self):
        """Test verifying incorrect password."""
        password = "my_secure_password"
        hashed = hash_password(password)

        assert verify_password("wrong_password", hashed) is False

    def test_different_hashes_for_same_password(self):
        """Test that same password produces different hashes (due to salt)."""
        password = "same_password"
        hash1 = hash_password(password)
        hash2 = hash_password(password)

        assert hash1 != hash2
        # But both should verify
        assert verify_password(password, hash1)
        assert verify_password(password, hash2)


class TestJWTHandler:
    """Tests for JWT token handling."""

    @pytest.fixture
    def config(self):
        return AuthConfig(jwt_secret="test_secret_key_12345")

    @pytest.fixture
    def handler(self, config):
        return JWTHandler(config)

    @pytest.fixture
    def user(self):
        return User(id="user-123", email="test@example.com", roles=["user"])

    def test_create_access_token(self, handler, user):
        """Test creating access token."""
        token = handler.create_access_token(user)

        assert isinstance(token, str)
        assert len(token) > 0

    def test_create_refresh_token(self, handler, user):
        """Test creating refresh token."""
        token = handler.create_refresh_token(user)

        assert isinstance(token, str)
        assert len(token) > 0

    def test_decode_valid_token(self, handler, user):
        """Test decoding valid token."""
        token = handler.create_access_token(user)
        payload = handler.decode_token(token)

        assert payload.sub == user.id
        assert payload.email == user.email
        assert payload.type == "access"
        assert payload.roles == user.roles

    def test_decode_refresh_token(self, handler, user):
        """Test decoding refresh token."""
        token = handler.create_refresh_token(user)
        payload = handler.decode_token(token)

        assert payload.type == "refresh"

    def test_decode_invalid_token(self, handler):
        """Test decoding invalid token raises error."""
        with pytest.raises(InvalidTokenError):
            handler.decode_token("invalid_token")

    def test_decode_expired_token(self, handler, user):
        """Test that expired tokens are rejected."""
        from datetime import timedelta

        # Create token that expired 1 hour ago
        token = handler._create_token(user, timedelta(hours=-1), "access")

        with pytest.raises(ExpiredTokenError):
            handler.decode_token(token)


class TestAPIKeyFunctions:
    """Tests for API key functions."""

    def test_generate_api_key(self):
        """Test generating API key."""
        key = generate_api_key()

        assert key.startswith("la_")
        assert len(key) > 10

    def test_generate_api_key_custom_prefix(self):
        """Test generating API key with custom prefix."""
        key = generate_api_key(prefix="custom_")

        assert key.startswith("custom_")

    def test_hash_api_key(self):
        """Test hashing API key."""
        key = "la_test_key_12345"
        hashed = hash_api_key(key)

        assert len(hashed) == 64  # SHA256

    def test_verify_api_key_correct(self):
        """Test verifying correct API key."""
        key = generate_api_key()
        hashed = hash_api_key(key)

        assert verify_api_key(key, hashed) is True

    def test_verify_api_key_incorrect(self):
        """Test verifying incorrect API key."""
        key = generate_api_key()
        hashed = hash_api_key(key)

        assert verify_api_key("wrong_key", hashed) is False


class TestAuthService:
    """Tests for AuthService."""

    @pytest.fixture
    def mock_db(self):
        """Create mock database session."""
        db = AsyncMock()
        db.add = MagicMock()
        db.commit = AsyncMock()
        db.refresh = AsyncMock()
        db.execute = AsyncMock()
        return db

    @pytest.fixture
    def service(self, mock_db):
        return AuthService(mock_db)

    @pytest.mark.asyncio
    async def test_register_new_user(self, service, mock_db):
        """Test registering a new user."""
        # Mock no existing user
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute.return_value = mock_result

        user = await service.register(
            email="new@example.com",
            password="password123",
            name="New User",
        )

        assert user.email == "new@example.com"
        assert user.name == "New User"
        mock_db.add.assert_called_once()
        mock_db.commit.assert_called()

    @pytest.mark.asyncio
    async def test_register_existing_user_raises(self, service, mock_db):
        """Test registering with existing email raises error."""
        from app.models.user import User as UserModel

        # Mock existing user
        existing = UserModel(id="123", email="existing@example.com", password_hash="hash")
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = existing
        mock_db.execute.return_value = mock_result

        with pytest.raises(UserExistsError):
            await service.register(
                email="existing@example.com",
                password="password123",
            )

    @pytest.mark.asyncio
    async def test_authenticate_success(self, service, mock_db):
        """Test successful authentication."""
        from app.models.user import User as UserModel

        password = "correct_password"
        hashed = hash_password(password)

        user = UserModel(
            id="user-1",
            email="test@example.com",
            password_hash=hashed,
            is_active=True,
        )
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = user
        mock_db.execute.return_value = mock_result

        result = await service.authenticate("test@example.com", password)

        assert result.email == "test@example.com"

    @pytest.mark.asyncio
    async def test_authenticate_wrong_password(self, service, mock_db):
        """Test authentication with wrong password."""
        from app.models.user import User as UserModel

        user = UserModel(
            id="user-1",
            email="test@example.com",
            password_hash=hash_password("correct_password"),
            is_active=True,
        )
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = user
        mock_db.execute.return_value = mock_result

        with pytest.raises(AuthenticationError):
            await service.authenticate("test@example.com", "wrong_password")

    @pytest.mark.asyncio
    async def test_authenticate_inactive_user(self, service, mock_db):
        """Test authentication with inactive user."""
        from app.models.user import User as UserModel

        password = "password123"
        user = UserModel(
            id="user-1",
            email="test@example.com",
            password_hash=hash_password(password),
            is_active=False,
        )
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = user
        mock_db.execute.return_value = mock_result

        with pytest.raises(AuthenticationError, match="disabled"):
            await service.authenticate("test@example.com", password)

    def test_create_tokens(self, service):
        """Test creating access and refresh tokens."""
        from app.models.user import User as UserModel

        user = UserModel(
            id="user-1",
            email="test@example.com",
            password_hash="hash",
            is_admin=False,
        )

        tokens = service.create_tokens(user)

        assert "access_token" in tokens
        assert "refresh_token" in tokens
        assert tokens["token_type"] == "bearer"


class TestTokenPayload:
    """Tests for TokenPayload."""

    def test_is_expired_false(self):
        """Test that non-expired token returns False."""
        from datetime import datetime, timezone, timedelta

        payload = TokenPayload(
            sub="user-1",
            email="test@example.com",
            exp=datetime.now(timezone.utc) + timedelta(hours=1),
        )

        assert payload.is_expired is False

    def test_is_expired_true(self):
        """Test that expired token returns True."""
        from datetime import datetime, timezone, timedelta

        payload = TokenPayload(
            sub="user-1",
            email="test@example.com",
            exp=datetime.now(timezone.utc) - timedelta(hours=1),
        )

        assert payload.is_expired is True
