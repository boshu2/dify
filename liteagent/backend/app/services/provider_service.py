from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.provider import LLMProvider
from app.schemas.provider import LLMProviderCreate, LLMProviderUpdate


class ProviderService:
    """Service for managing LLM providers."""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def create(self, data: LLMProviderCreate) -> LLMProvider:
        provider = LLMProvider(
            name=data.name,
            provider_type=data.provider_type.value,
            api_key=data.api_key,
            base_url=data.base_url,
            model_name=data.model_name,
        )
        self.db.add(provider)
        await self.db.commit()
        await self.db.refresh(provider)
        return provider

    async def get_by_id(self, provider_id: str) -> LLMProvider | None:
        result = await self.db.execute(
            select(LLMProvider).where(LLMProvider.id == provider_id)
        )
        return result.scalar_one_or_none()

    async def get_all(self, active_only: bool = False) -> list[LLMProvider]:
        query = select(LLMProvider)
        if active_only:
            query = query.where(LLMProvider.is_active.is_(True))
        query = query.order_by(LLMProvider.created_at.desc())
        result = await self.db.execute(query)
        return list(result.scalars().all())

    async def update(self, provider_id: str, data: LLMProviderUpdate) -> LLMProvider | None:
        provider = await self.get_by_id(provider_id)
        if not provider:
            return None

        update_data = data.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            setattr(provider, field, value)

        await self.db.commit()
        await self.db.refresh(provider)
        return provider

    async def delete(self, provider_id: str) -> bool:
        provider = await self.get_by_id(provider_id)
        if not provider:
            return False

        await self.db.delete(provider)
        await self.db.commit()
        return True
