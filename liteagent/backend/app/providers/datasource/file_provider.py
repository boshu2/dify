from pathlib import Path

import aiofiles

from app.extractors.registry import get_extractor, get_registry
from app.providers.datasource.base import BaseDataSourceProvider, DataSourceContent


class FileDataSourceProvider(BaseDataSourceProvider):
    """Provider for local file data sources."""

    # Text files that can be read directly
    TEXT_EXTENSIONS = {".txt", ".md", ".json", ".xml", ".yaml", ".yml", ".py", ".js", ".ts", ".css"}

    def __init__(self, upload_dir: str = "./uploads"):
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)

    @property
    def supported_extensions(self) -> set[str]:
        """All supported extensions (text + extractable documents)."""
        return self.TEXT_EXTENSIONS | get_registry().supported_extensions()

    async def fetch_content(self, source: str) -> DataSourceContent:
        file_path = self.upload_dir / source
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {source}")

        ext = file_path.suffix.lower()
        metadata = {
            "filename": file_path.name,
            "size": file_path.stat().st_size,
            "extension": ext,
        }

        # Check if we have an extractor for this file type
        extractor = get_extractor(file_path.name)
        if extractor:
            # Use extractor for binary documents
            extracted = await extractor.extract_from_path(file_path)
            metadata.update(extracted.metadata)
            metadata["pages"] = extracted.pages
            metadata["word_count"] = extracted.word_count
            return DataSourceContent(
                content=extracted.text,
                source=str(file_path),
                metadata=metadata,
            )

        # Fallback to text reading for text files
        async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
            content = await f.read()

        return DataSourceContent(
            content=content,
            source=str(file_path),
            metadata=metadata,
        )

    async def validate_source(self, source: str) -> bool:
        file_path = self.upload_dir / source
        return file_path.exists() and file_path.suffix.lower() in self.supported_extensions

    async def save_uploaded_file(self, filename: str, content: bytes) -> str:
        """Save an uploaded file and return the path."""
        file_path = self.upload_dir / filename
        async with aiofiles.open(file_path, "wb") as f:
            await f.write(content)
        return filename
