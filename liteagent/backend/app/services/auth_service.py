"""Authentication service for user management."""
import hashlib
import secrets
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth import (
    AuthConfig,
    JWTHandler,
    generate_api_key,
    hash_api_key,
)
from app.models.user import APIKey, User


class AuthenticationError(Exception):
    """Authentication failed."""
    pass


class UserExistsError(Exception):
    """User already exists."""
    pass


def hash_password(password: str) -> str:
    """Hash password using SHA256 with salt."""
    salt = secrets.token_hex(16)
    hashed = hashlib.sha256((salt + password).encode()).hexdigest()
    return f"{salt}:{hashed}"


def verify_password(password: str, stored_hash: str) -> bool:
    """Verify password against stored hash."""
    try:
        salt, hashed = stored_hash.split(":")
        check_hash = hashlib.sha256((salt + password).encode()).hexdigest()
        return secrets.compare_digest(hashed, check_hash)
    except ValueError:
        return False


class AuthService:
    """Service for authentication operations."""

    def __init__(self, db: AsyncSession, config: AuthConfig | None = None):
        self.db = db
        self.config = config or AuthConfig.from_env()
        self.jwt_handler = JWTHandler(self.config)

    async def register(
        self,
        email: str,
        password: str,
        name: str | None = None,
    ) -> User:
        """Register a new user."""
        # Check if user exists
        existing = await self.get_user_by_email(email)
        if existing:
            raise UserExistsError(f"User with email {email} already exists")

        # Create user
        user = User(
            email=email.lower().strip(),
            password_hash=hash_password(password),
            name=name,
        )

        self.db.add(user)
        await self.db.commit()
        await self.db.refresh(user)

        return user

    async def authenticate(self, email: str, password: str) -> User:
        """Authenticate user with email and password."""
        user = await self.get_user_by_email(email)

        if not user:
            raise AuthenticationError("Invalid email or password")

        if not verify_password(password, user.password_hash):
            raise AuthenticationError("Invalid email or password")

        if not user.is_active:
            raise AuthenticationError("User account is disabled")

        # Update last login
        user.last_login_at = datetime.now(timezone.utc)
        await self.db.commit()

        return user

    async def get_user_by_email(self, email: str) -> User | None:
        """Get user by email address."""
        result = await self.db.execute(
            select(User).where(User.email == email.lower().strip())
        )
        return result.scalar_one_or_none()

    async def get_user_by_id(self, user_id: str) -> User | None:
        """Get user by ID."""
        result = await self.db.execute(
            select(User).where(User.id == user_id)
        )
        return result.scalar_one_or_none()

    def create_tokens(self, user: User) -> dict:
        """Create access and refresh tokens for user."""
        from app.core.auth import User as AuthUser

        auth_user = AuthUser(
            id=user.id,
            email=user.email,
            roles=["admin"] if user.is_admin else ["user"],
        )

        return {
            "access_token": self.jwt_handler.create_access_token(auth_user),
            "refresh_token": self.jwt_handler.create_refresh_token(auth_user),
            "token_type": "bearer",
        }

    async def refresh_token(self, refresh_token: str) -> dict:
        """Refresh access token using refresh token."""
        from app.core.auth import ExpiredTokenError, InvalidTokenError

        try:
            payload = self.jwt_handler.decode_token(refresh_token)
        except (ExpiredTokenError, InvalidTokenError) as e:
            raise AuthenticationError(str(e))

        if payload.type != "refresh":
            raise AuthenticationError("Invalid token type")

        user = await self.get_user_by_id(payload.sub)
        if not user or not user.is_active:
            raise AuthenticationError("User not found or disabled")

        return self.create_tokens(user)

    async def create_api_key(
        self,
        user_id: str,
        name: str,
        expires_in_days: int | None = None,
    ) -> tuple[str, APIKey]:
        """Create a new API key for user. Returns (raw_key, api_key_model)."""
        raw_key = generate_api_key()
        key_hash = hash_api_key(raw_key)
        key_prefix = raw_key[:12]

        expires_at = None
        if expires_in_days:
            from datetime import timedelta
            expires_at = datetime.now(timezone.utc) + timedelta(days=expires_in_days)

        api_key = APIKey(
            user_id=user_id,
            name=name,
            key_hash=key_hash,
            key_prefix=key_prefix,
            expires_at=expires_at,
        )

        self.db.add(api_key)
        await self.db.commit()
        await self.db.refresh(api_key)

        return raw_key, api_key

    async def get_api_keys(self, user_id: str) -> list[APIKey]:
        """Get all API keys for a user."""
        result = await self.db.execute(
            select(APIKey).where(APIKey.user_id == user_id)
        )
        return list(result.scalars().all())

    async def revoke_api_key(self, key_id: str, user_id: str) -> bool:
        """Revoke an API key."""
        result = await self.db.execute(
            select(APIKey).where(
                APIKey.id == key_id,
                APIKey.user_id == user_id,
            )
        )
        api_key = result.scalar_one_or_none()

        if not api_key:
            return False

        api_key.is_active = False
        await self.db.commit()
        return True

    async def validate_api_key(self, raw_key: str) -> User | None:
        """Validate API key and return associated user."""
        key_hash = hash_api_key(raw_key)

        result = await self.db.execute(
            select(APIKey).where(
                APIKey.key_hash == key_hash,
                APIKey.is_active == True,
            )
        )
        api_key = result.scalar_one_or_none()

        if not api_key:
            return None

        if api_key.is_expired:
            return None

        # Update last used
        api_key.last_used_at = datetime.now(timezone.utc)
        await self.db.commit()

        return await self.get_user_by_id(api_key.user_id)
