from app.providers.datasource.base import BaseDataSourceProvider
from app.providers.datasource.file_provider import FileDataSourceProvider
from app.providers.datasource.url_provider import URLDataSourceProvider
from app.providers.datasource.text_provider import TextDataSourceProvider
from app.schemas.datasource import DataSourceType


class DataSourceFactory:
    """Factory for creating data source provider instances."""

    _providers: dict[DataSourceType, type[BaseDataSourceProvider]] = {
        DataSourceType.FILE: FileDataSourceProvider,
        DataSourceType.URL: URLDataSourceProvider,
        DataSourceType.TEXT: TextDataSourceProvider,
    }

    @classmethod
    def create(cls, source_type: DataSourceType, **kwargs) -> BaseDataSourceProvider:
        """Create a data source provider instance."""
        provider_class = cls._providers.get(source_type)
        if not provider_class:
            raise ValueError(f"Unknown data source type: {source_type}")

        return provider_class(**kwargs)

    @classmethod
    def get_all_types(cls) -> list[DataSourceType]:
        """Get all supported data source types."""
        return list(cls._providers.keys())
