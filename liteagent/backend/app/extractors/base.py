"""
Base document extractor interface.

Document extractors convert binary file formats to plain text for RAG processing.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO


@dataclass
class ExtractedContent:
    """Extracted content from a document."""
    text: str
    metadata: dict = field(default_factory=dict)
    pages: int = 1
    word_count: int = 0

    def __post_init__(self):
        if self.word_count == 0 and self.text:
            self.word_count = len(self.text.split())


class DocumentExtractor(ABC):
    """Abstract base class for document extractors."""

    @property
    @abstractmethod
    def supported_extensions(self) -> set[str]:
        """Return set of supported file extensions (lowercase, with dot)."""
        pass

    @abstractmethod
    async def extract(self, file: BinaryIO, filename: str) -> ExtractedContent:
        """
        Extract text content from a binary file.

        Args:
            file: Binary file-like object.
            filename: Original filename (for metadata).

        Returns:
            ExtractedContent with extracted text and metadata.
        """
        pass

    def can_extract(self, filename: str) -> bool:
        """Check if this extractor can handle the given file."""
        ext = Path(filename).suffix.lower()
        return ext in self.supported_extensions

    async def extract_from_path(self, file_path: str | Path) -> ExtractedContent:
        """Extract content from a file path."""
        path = Path(file_path)
        with open(path, "rb") as f:
            return await self.extract(f, path.name)
