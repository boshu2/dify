"""
HTML document extractor using BeautifulSoup.

Extracts text content from HTML files for RAG processing.
"""
from typing import BinaryIO

from app.extractors.base import DocumentExtractor, ExtractedContent


class HTMLExtractor(DocumentExtractor):
    """Extract text from HTML documents using BeautifulSoup."""

    # Tags to remove completely (including their content)
    REMOVE_TAGS = {"script", "style", "nav", "footer", "header", "aside", "noscript"}

    @property
    def supported_extensions(self) -> set[str]:
        return {".html", ".htm", ".xhtml"}

    async def extract(self, file: BinaryIO, filename: str) -> ExtractedContent:
        """Extract text from HTML file."""
        from bs4 import BeautifulSoup

        content = file.read()

        # Try to decode as UTF-8, fallback to latin-1
        try:
            html_text = content.decode("utf-8")
        except UnicodeDecodeError:
            html_text = content.decode("latin-1")

        soup = BeautifulSoup(html_text, "lxml")
        metadata = {
            "filename": filename,
            "format": "html",
        }

        # Extract title
        if soup.title and soup.title.string:
            metadata["title"] = soup.title.string.strip()

        # Extract meta description
        meta_desc = soup.find("meta", attrs={"name": "description"})
        if meta_desc and meta_desc.get("content"):
            metadata["description"] = meta_desc["content"]

        # Remove unwanted tags
        for tag in self.REMOVE_TAGS:
            for element in soup.find_all(tag):
                element.decompose()

        # Get text with preserved structure
        # Use get_text with separator to preserve some structure
        text = soup.get_text(separator="\n", strip=True)

        # Clean up multiple newlines
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        full_text = "\n\n".join(lines)

        return ExtractedContent(
            text=full_text,
            metadata=metadata,
            pages=1,
        )
