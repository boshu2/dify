from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.datasource import DataSource, DataSourceType
from app.schemas.datasource import DataSourceCreate, DataSourceUpdate
from app.schemas.datasource import DataSourceType as SchemaDataSourceType
from app.providers.datasource import DataSourceFactory


class DataSourceService:
    """Service for managing data sources."""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def create(self, data: DataSourceCreate) -> DataSource:
        content = data.content
        source_type = DataSourceType(data.source_type.value)

        # Fetch content based on source type
        if source_type == DataSourceType.URL and data.source_path:
            provider = DataSourceFactory.create(SchemaDataSourceType.URL)
            result = await provider.fetch_content(data.source_path)
            content = result.content

        elif source_type == DataSourceType.GITLAB and data.source_path:
            # Create GitLab provider with credentials
            provider = DataSourceFactory.create(
                SchemaDataSourceType.GITLAB,
                gitlab_url=data.gitlab_url or "https://gitlab.com",
                access_token=data.gitlab_token or "",
            )
            result = await provider.fetch_content(data.source_path)
            content = result.content

        datasource = DataSource(
            name=data.name,
            source_type=source_type,
            content=content,
            source_path=data.source_path,
            gitlab_url=data.gitlab_url if source_type == DataSourceType.GITLAB else None,
            gitlab_token=data.gitlab_token if source_type == DataSourceType.GITLAB else None,
        )
        self.db.add(datasource)
        await self.db.commit()
        await self.db.refresh(datasource)
        return datasource

    async def get_by_id(self, datasource_id: str) -> DataSource | None:
        result = await self.db.execute(
            select(DataSource).where(DataSource.id == datasource_id)
        )
        return result.scalar_one_or_none()

    async def get_all(self) -> list[DataSource]:
        result = await self.db.execute(
            select(DataSource).order_by(DataSource.created_at.desc())
        )
        return list(result.scalars().all())

    async def get_by_ids(self, ids: list[str]) -> list[DataSource]:
        if not ids:
            return []
        result = await self.db.execute(
            select(DataSource).where(DataSource.id.in_(ids))
        )
        return list(result.scalars().all())

    async def update(self, datasource_id: str, data: DataSourceUpdate) -> DataSource | None:
        datasource = await self.get_by_id(datasource_id)
        if not datasource:
            return None

        update_data = data.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            setattr(datasource, field, value)

        await self.db.commit()
        await self.db.refresh(datasource)
        return datasource

    async def delete(self, datasource_id: str) -> bool:
        datasource = await self.get_by_id(datasource_id)
        if not datasource:
            return False

        await self.db.delete(datasource)
        await self.db.commit()
        return True

    async def refresh_content(self, datasource_id: str) -> DataSource | None:
        """Re-fetch content for URL or GitLab data sources."""
        datasource = await self.get_by_id(datasource_id)
        if not datasource:
            return None

        if datasource.source_type == DataSourceType.URL:
            provider = DataSourceFactory.create(SchemaDataSourceType.URL)
            result = await provider.fetch_content(datasource.source_path)
            datasource.content = result.content

        elif datasource.source_type == DataSourceType.GITLAB:
            provider = DataSourceFactory.create(
                SchemaDataSourceType.GITLAB,
                gitlab_url=datasource.gitlab_url or "https://gitlab.com",
                access_token=datasource.gitlab_token or "",
            )
            result = await provider.fetch_content(datasource.source_path)
            datasource.content = result.content

        else:
            return None

        await self.db.commit()
        await self.db.refresh(datasource)
        return datasource
