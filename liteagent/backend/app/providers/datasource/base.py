from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class DataSourceContent:
    content: str
    source: str
    metadata: dict | None = None


class BaseDataSourceProvider(ABC):
    """Base class for data source providers."""

    @abstractmethod
    async def fetch_content(self, source: str) -> DataSourceContent:
        """Fetch content from the data source."""
        pass

    @abstractmethod
    async def validate_source(self, source: str) -> bool:
        """Validate that the source is accessible."""
        pass
