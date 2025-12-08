"""
PDF document extractor using pypdf.

Extracts text content from PDF files for RAG processing.
"""
import io
from typing import BinaryIO

from app.extractors.base import DocumentExtractor, ExtractedContent


class PDFExtractor(DocumentExtractor):
    """Extract text from PDF documents using pypdf."""

    @property
    def supported_extensions(self) -> set[str]:
        return {".pdf"}

    async def extract(self, file: BinaryIO, filename: str) -> ExtractedContent:
        """Extract text from PDF file."""
        from pypdf import PdfReader

        # Read file content into BytesIO for pypdf
        content = file.read()
        pdf_file = io.BytesIO(content)

        reader = PdfReader(pdf_file)
        pages_text = []
        metadata = {
            "filename": filename,
            "format": "pdf",
        }

        # Extract PDF metadata if available
        if reader.metadata:
            if reader.metadata.title:
                metadata["title"] = reader.metadata.title
            if reader.metadata.author:
                metadata["author"] = reader.metadata.author
            if reader.metadata.subject:
                metadata["subject"] = reader.metadata.subject
            if reader.metadata.creator:
                metadata["creator"] = reader.metadata.creator

        # Extract text from each page
        for page_num, page in enumerate(reader.pages, 1):
            text = page.extract_text()
            if text:
                pages_text.append(text.strip())

        full_text = "\n\n".join(pages_text)

        return ExtractedContent(
            text=full_text,
            metadata=metadata,
            pages=len(reader.pages),
        )
