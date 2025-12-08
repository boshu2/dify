"""
Text utilities for RAG pipeline.

Provides text normalization, cleaning, and preprocessing for better retrieval.
"""
import re
import unicodedata
from typing import Callable


class TextNormalizer:
    """
    Text normalizer for preprocessing documents before embedding.

    Provides consistent text cleaning and normalization to improve
    embedding quality and retrieval accuracy.
    """

    def __init__(
        self,
        lowercase: bool = False,
        remove_extra_whitespace: bool = True,
        normalize_unicode: bool = True,
        remove_urls: bool = False,
        remove_emails: bool = False,
        remove_html_tags: bool = True,
        min_word_length: int = 0,
        max_word_length: int = 50,
        custom_processors: list[Callable[[str], str]] | None = None,
    ):
        """
        Initialize text normalizer.

        Args:
            lowercase: Convert text to lowercase.
            remove_extra_whitespace: Collapse multiple whitespace to single.
            normalize_unicode: Normalize unicode characters (NFKC).
            remove_urls: Remove URLs from text.
            remove_emails: Remove email addresses.
            remove_html_tags: Remove HTML tags.
            min_word_length: Filter words shorter than this.
            max_word_length: Filter words longer than this.
            custom_processors: Additional processing functions.
        """
        self.lowercase = lowercase
        self.remove_extra_whitespace = remove_extra_whitespace
        self.normalize_unicode = normalize_unicode
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
        self.remove_html_tags = remove_html_tags
        self.min_word_length = min_word_length
        self.max_word_length = max_word_length
        self.custom_processors = custom_processors or []

        # Precompile patterns
        self._url_pattern = re.compile(
            r'https?://\S+|www\.\S+',
            re.IGNORECASE
        )
        self._email_pattern = re.compile(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        )
        self._html_pattern = re.compile(r'<[^>]+>')
        self._whitespace_pattern = re.compile(r'\s+')

    def normalize(self, text: str) -> str:
        """
        Apply all normalization steps to text.

        Args:
            text: Input text to normalize.

        Returns:
            Normalized text.
        """
        if not text:
            return ""

        # Unicode normalization
        if self.normalize_unicode:
            text = unicodedata.normalize('NFKC', text)

        # Remove HTML tags
        if self.remove_html_tags:
            text = self._html_pattern.sub(' ', text)

        # Remove URLs
        if self.remove_urls:
            text = self._url_pattern.sub(' ', text)

        # Remove emails
        if self.remove_emails:
            text = self._email_pattern.sub(' ', text)

        # Lowercase
        if self.lowercase:
            text = text.lower()

        # Remove extra whitespace
        if self.remove_extra_whitespace:
            text = self._whitespace_pattern.sub(' ', text)
            text = text.strip()

        # Filter by word length
        if self.min_word_length > 0 or self.max_word_length < 50:
            words = text.split()
            filtered = [
                w for w in words
                if self.min_word_length <= len(w) <= self.max_word_length
            ]
            text = ' '.join(filtered)

        # Apply custom processors
        for processor in self.custom_processors:
            text = processor(text)

        return text

    def normalize_for_embedding(self, text: str) -> str:
        """
        Normalize text optimized for embedding generation.

        Preserves more structure than aggressive normalization.
        """
        if not text:
            return ""

        # Unicode normalization
        text = unicodedata.normalize('NFKC', text)

        # Remove HTML tags
        text = self._html_pattern.sub(' ', text)

        # Normalize whitespace but preserve paragraph structure
        lines = text.split('\n')
        normalized_lines = []
        for line in lines:
            line = self._whitespace_pattern.sub(' ', line).strip()
            if line:
                normalized_lines.append(line)

        return '\n'.join(normalized_lines)

    def normalize_for_search(self, text: str) -> str:
        """
        Normalize text optimized for search queries.

        More aggressive normalization for matching.
        """
        if not text:
            return ""

        # Full normalization
        text = unicodedata.normalize('NFKC', text)
        text = self._html_pattern.sub(' ', text)
        text = self._url_pattern.sub(' ', text)
        text = text.lower()
        text = self._whitespace_pattern.sub(' ', text).strip()

        return text


def extract_sentences(text: str) -> list[str]:
    """
    Extract sentences from text.

    Uses simple sentence boundary detection that handles common cases.
    """
    if not text:
        return []

    # Split on sentence boundaries
    sentence_pattern = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')
    sentences = sentence_pattern.split(text)

    # Clean up and filter
    result = []
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence and len(sentence) > 10:  # Filter very short "sentences"
            result.append(sentence)

    return result


def extract_paragraphs(text: str) -> list[str]:
    """
    Extract paragraphs from text.

    Splits on double newlines or similar paragraph markers.
    """
    if not text:
        return []

    # Split on paragraph boundaries
    paragraphs = re.split(r'\n\s*\n', text)

    # Clean up and filter
    result = []
    for para in paragraphs:
        para = para.strip()
        if para and len(para) > 20:  # Filter very short paragraphs
            result.append(para)

    return result


def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """
    Truncate text to max length, trying to break at word boundary.

    Args:
        text: Text to truncate.
        max_length: Maximum length including suffix.
        suffix: Suffix to add if truncated.

    Returns:
        Truncated text.
    """
    if not text or len(text) <= max_length:
        return text

    # Find last space before max_length
    truncate_at = max_length - len(suffix)
    last_space = text.rfind(' ', 0, truncate_at)

    if last_space > truncate_at // 2:
        # Use word boundary
        return text[:last_space] + suffix
    else:
        # No good word boundary, just cut
        return text[:truncate_at] + suffix


def count_tokens_approximate(text: str) -> int:
    """
    Approximate token count for text.

    Uses simple word/character heuristics. For exact counts, use tiktoken.

    Args:
        text: Input text.

    Returns:
        Approximate token count.
    """
    if not text:
        return 0

    # Rough approximation: ~4 characters per token for English
    # Words are roughly 1.3 tokens on average
    words = len(text.split())
    chars = len(text)

    # Use a weighted average
    return int(words * 1.3 + chars / 4) // 2


def clean_markdown(text: str) -> str:
    """
    Remove markdown formatting from text while preserving content.

    Args:
        text: Markdown text.

    Returns:
        Plain text.
    """
    if not text:
        return ""

    # Remove code blocks
    text = re.sub(r'```[\s\S]*?```', '', text)
    text = re.sub(r'`[^`]+`', '', text)

    # Remove headers
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)

    # Remove bold/italic
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'\*([^*]+)\*', r'\1', text)
    text = re.sub(r'__([^_]+)__', r'\1', text)
    text = re.sub(r'_([^_]+)_', r'\1', text)

    # Remove links but keep text
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)

    # Remove images
    text = re.sub(r'!\[([^\]]*)\]\([^)]+\)', '', text)

    # Remove horizontal rules
    text = re.sub(r'^[-*_]{3,}\s*$', '', text, flags=re.MULTILINE)

    # Remove list markers
    text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)

    # Clean up whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = text.strip()

    return text


# Default normalizer instance
default_normalizer = TextNormalizer()
