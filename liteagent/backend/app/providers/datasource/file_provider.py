from pathlib import Path

import aiofiles

from app.providers.datasource.base import BaseDataSourceProvider, DataSourceContent


class FileDataSourceProvider(BaseDataSourceProvider):
    """Provider for local file data sources."""

    SUPPORTED_EXTENSIONS = {".txt", ".md", ".json", ".csv", ".xml", ".yaml", ".yml", ".py", ".js"}

    def __init__(self, upload_dir: str = "./uploads"):
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)

    async def fetch_content(self, source: str) -> DataSourceContent:
        file_path = self.upload_dir / source
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {source}")

        async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
            content = await f.read()

        return DataSourceContent(
            content=content,
            source=str(file_path),
            metadata={
                "filename": file_path.name,
                "size": file_path.stat().st_size,
                "extension": file_path.suffix,
            },
        )

    async def validate_source(self, source: str) -> bool:
        file_path = self.upload_dir / source
        return file_path.exists() and file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS

    async def save_uploaded_file(self, filename: str, content: bytes) -> str:
        """Save an uploaded file and return the path."""
        file_path = self.upload_dir / filename
        async with aiofiles.open(file_path, "wb") as f:
            await f.write(content)
        return filename
