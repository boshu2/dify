from app.providers.datasource.base import BaseDataSourceProvider, DataSourceContent


class TextDataSourceProvider(BaseDataSourceProvider):
    """Provider for plain text data sources (directly provided content)."""

    async def fetch_content(self, source: str) -> DataSourceContent:
        # For text sources, the "source" is actually the content itself
        return DataSourceContent(
            content=source,
            source="inline_text",
            metadata={
                "type": "inline",
                "length": len(source),
            },
        )

    async def validate_source(self, source: str) -> bool:
        # Text is always valid if it's a non-empty string
        return bool(source and source.strip())
