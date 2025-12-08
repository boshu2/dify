"""
Base document extractor interface.

Document extractors convert binary file formats to plain text for RAG processing.
"""
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import BinaryIO

logger = logging.getLogger(__name__)


class ExtractionError(Exception):
    """Base exception for extraction errors."""
    pass


class UnsupportedFormatError(ExtractionError):
    """Raised when file format is not supported."""
    pass


class CorruptedFileError(ExtractionError):
    """Raised when file appears to be corrupted."""
    pass


class ExtractionStatus(str, Enum):
    """Status of extraction operation."""
    SUCCESS = "success"
    PARTIAL = "partial"  # Some content extracted but with issues
    FAILED = "failed"


@dataclass
class ExtractedContent:
    """Extracted content from a document."""
    text: str
    metadata: dict = field(default_factory=dict)
    pages: int = 1
    word_count: int = 0
    status: ExtractionStatus = ExtractionStatus.SUCCESS
    warnings: list[str] = field(default_factory=list)

    def __post_init__(self):
        if self.word_count == 0 and self.text:
            self.word_count = len(self.text.split())

    @property
    def is_successful(self) -> bool:
        """Check if extraction was successful."""
        return self.status in (ExtractionStatus.SUCCESS, ExtractionStatus.PARTIAL)

    def add_warning(self, warning: str) -> None:
        """Add a warning message."""
        self.warnings.append(warning)
        if self.status == ExtractionStatus.SUCCESS:
            self.status = ExtractionStatus.PARTIAL


class DocumentExtractor(ABC):
    """
    Abstract base class for document extractors.

    Provides common functionality for extracting text from binary documents.
    """

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

        Raises:
            ExtractionError: If extraction fails.
        """
        pass

    def can_extract(self, filename: str) -> bool:
        """Check if this extractor can handle the given file."""
        ext = Path(filename).suffix.lower()
        return ext in self.supported_extensions

    async def extract_from_path(self, file_path: str | Path) -> ExtractedContent:
        """
        Extract content from a file path.

        Args:
            file_path: Path to the file.

        Returns:
            ExtractedContent with extracted text and metadata.

        Raises:
            FileNotFoundError: If file doesn't exist.
            ExtractionError: If extraction fails.
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if not path.is_file():
            raise ExtractionError(f"Not a file: {file_path}")

        try:
            with open(path, "rb") as f:
                return await self.extract(f, path.name)
        except ExtractionError:
            raise
        except Exception as e:
            logger.exception(f"Unexpected error extracting {path.name}")
            raise ExtractionError(f"Failed to extract {path.name}: {e}") from e

    async def safe_extract(
        self,
        file: BinaryIO,
        filename: str,
    ) -> ExtractedContent:
        """
        Extract content with error handling - never raises.

        Args:
            file: Binary file-like object.
            filename: Original filename.

        Returns:
            ExtractedContent, with status=FAILED if extraction failed.
        """
        try:
            return await self.extract(file, filename)
        except Exception as e:
            logger.warning(f"Extraction failed for {filename}: {e}")
            return ExtractedContent(
                text="",
                metadata={"filename": filename, "error": str(e)},
                status=ExtractionStatus.FAILED,
                warnings=[f"Extraction failed: {e}"],
            )


def validate_file_size(
    file: BinaryIO,
    max_size_mb: float = 100,
) -> bool:
    """
    Validate file size before extraction.

    Args:
        file: Binary file object.
        max_size_mb: Maximum allowed size in megabytes.

    Returns:
        True if file is within size limit.

    Raises:
        ExtractionError: If file is too large.
    """
    current_pos = file.tell()
    file.seek(0, 2)  # Seek to end
    size = file.tell()
    file.seek(current_pos)  # Restore position

    max_bytes = max_size_mb * 1024 * 1024
    if size > max_bytes:
        raise ExtractionError(
            f"File too large: {size / (1024*1024):.1f}MB > {max_size_mb}MB limit"
        )

    return True
