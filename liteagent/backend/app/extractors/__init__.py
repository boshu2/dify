"""
Document extractors for RAG pipeline.

Supports extraction from:
- PDF (.pdf)
- Word (.docx)
- HTML (.html, .htm, .xhtml)
- Excel (.xlsx, .xls)
- CSV (.csv)
"""
from app.extractors.base import (
    DocumentExtractor,
    ExtractedContent,
    ExtractionError,
    ExtractionStatus,
    CorruptedFileError,
    UnsupportedFormatError,
    validate_file_size,
)
from app.extractors.docx import DocxExtractor
from app.extractors.html import HTMLExtractor
from app.extractors.pdf import PDFExtractor
from app.extractors.registry import ExtractorRegistry, get_extractor
from app.extractors.spreadsheet import SpreadsheetExtractor

__all__ = [
    "DocumentExtractor",
    "ExtractedContent",
    "ExtractionError",
    "ExtractionStatus",
    "CorruptedFileError",
    "UnsupportedFormatError",
    "validate_file_size",
    "PDFExtractor",
    "DocxExtractor",
    "HTMLExtractor",
    "SpreadsheetExtractor",
    "ExtractorRegistry",
    "get_extractor",
]
