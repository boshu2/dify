"""Authentication schemas."""
from pydantic import BaseModel, EmailStr, Field


class UserRegister(BaseModel):
    """User registration request."""
    email: EmailStr
    password: str = Field(..., min_length=8)
    name: str | None = None


class UserLogin(BaseModel):
    """User login request."""
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    """Token response."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class TokenRefresh(BaseModel):
    """Token refresh request."""
    refresh_token: str


class UserResponse(BaseModel):
    """User response (public fields only)."""
    id: str
    email: str
    name: str | None
    is_active: bool
    is_admin: bool
    created_at: str | None

    class Config:
        from_attributes = True


class APIKeyCreate(BaseModel):
    """API key creation request."""
    name: str = Field(..., min_length=1, max_length=100)
    expires_in_days: int | None = Field(None, ge=1, le=365)


class APIKeyResponse(BaseModel):
    """API key response."""
    id: str
    name: str
    key_prefix: str
    is_active: bool
    last_used_at: str | None
    expires_at: str | None
    created_at: str | None

    class Config:
        from_attributes = True


class APIKeyCreated(BaseModel):
    """Response when API key is created (includes raw key)."""
    key: str  # Only shown once!
    api_key: APIKeyResponse
