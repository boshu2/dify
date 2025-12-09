from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.exceptions import ProviderNotFoundError
from app.schemas.provider import (
    LLMProviderCreate,
    LLMProviderResponse,
    LLMProviderUpdate,
)
from app.services.provider_service import ProviderService

router = APIRouter()


@router.post("/", response_model=LLMProviderResponse)
async def create_provider(
    data: LLMProviderCreate,
    db: AsyncSession = Depends(get_db),
):
    """Create a new LLM provider configuration."""
    service = ProviderService(db)
    provider = await service.create(data)
    return provider


@router.get("/", response_model=list[LLMProviderResponse])
async def list_providers(
    active_only: bool = False,
    db: AsyncSession = Depends(get_db),
):
    """List all LLM provider configurations."""
    service = ProviderService(db)
    providers = await service.get_all(active_only=active_only)
    return providers


@router.get("/{provider_id}", response_model=LLMProviderResponse)
async def get_provider(
    provider_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Get a specific LLM provider configuration."""
    service = ProviderService(db)
    provider = await service.get_by_id(provider_id)
    if not provider:
        raise ProviderNotFoundError(provider_id).to_http_exception()
    return provider


@router.patch("/{provider_id}", response_model=LLMProviderResponse)
async def update_provider(
    provider_id: str,
    data: LLMProviderUpdate,
    db: AsyncSession = Depends(get_db),
):
    """Update an LLM provider configuration."""
    service = ProviderService(db)
    provider = await service.update(provider_id, data)
    if not provider:
        raise ProviderNotFoundError(provider_id).to_http_exception()
    return provider


@router.delete("/{provider_id}")
async def delete_provider(
    provider_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Delete an LLM provider configuration."""
    service = ProviderService(db)
    success = await service.delete(provider_id)
    if not success:
        raise ProviderNotFoundError(provider_id).to_http_exception()
    return {"status": "deleted"}
