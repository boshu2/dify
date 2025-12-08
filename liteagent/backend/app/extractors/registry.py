"""
Extractor registry for document type resolution.

Automatically selects the appropriate extractor based on file extension.
"""
from pathlib import Path

from app.extractors.base import DocumentExtractor
from app.extractors.docx import DocxExtractor
from app.extractors.html import HTMLExtractor
from app.extractors.pdf import PDFExtractor
from app.extractors.spreadsheet import SpreadsheetExtractor


class ExtractorRegistry:
    """Registry of document extractors."""

    def __init__(self):
        self._extractors: list[DocumentExtractor] = []
        self._extension_map: dict[str, DocumentExtractor] = {}

    def register(self, extractor: DocumentExtractor) -> None:
        """Register an extractor."""
        self._extractors.append(extractor)
        for ext in extractor.supported_extensions:
            self._extension_map[ext.lower()] = extractor

    def get_extractor(self, filename: str) -> DocumentExtractor | None:
        """Get extractor for a filename based on extension."""
        ext = Path(filename).suffix.lower()
        return self._extension_map.get(ext)

    def supported_extensions(self) -> set[str]:
        """Return all supported extensions."""
        return set(self._extension_map.keys())

    def can_extract(self, filename: str) -> bool:
        """Check if any extractor can handle this file."""
        return self.get_extractor(filename) is not None


# Default registry with all built-in extractors
_default_registry = ExtractorRegistry()
_default_registry.register(PDFExtractor())
_default_registry.register(DocxExtractor())
_default_registry.register(HTMLExtractor())
_default_registry.register(SpreadsheetExtractor())


def get_extractor(filename: str) -> DocumentExtractor | None:
    """Get extractor from default registry."""
    return _default_registry.get_extractor(filename)


def get_registry() -> ExtractorRegistry:
    """Get the default extractor registry."""
    return _default_registry
