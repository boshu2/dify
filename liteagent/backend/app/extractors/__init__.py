"""
Document extractors for RAG pipeline.

Supports extraction from:
- PDF (.pdf)
- Word (.docx)
- HTML (.html, .htm, .xhtml)
- Excel (.xlsx, .xls)
- CSV (.csv)
"""
from app.extractors.base import DocumentExtractor, ExtractedContent
from app.extractors.docx import DocxExtractor
from app.extractors.html import HTMLExtractor
from app.extractors.pdf import PDFExtractor
from app.extractors.registry import ExtractorRegistry, get_extractor
from app.extractors.spreadsheet import SpreadsheetExtractor

__all__ = [
    "DocumentExtractor",
    "ExtractedContent",
    "PDFExtractor",
    "DocxExtractor",
    "HTMLExtractor",
    "SpreadsheetExtractor",
    "ExtractorRegistry",
    "get_extractor",
]
