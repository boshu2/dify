"""
Text chunking strategies for RAG pipeline.

Supports:
- Fixed-size chunking with overlap
- Semantic chunking (sentence-aware)
- Recursive chunking for structured documents
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
import re
import uuid

from app.rag.vector_store import Document


@dataclass
class Chunk:
    """A text chunk with metadata."""
    content: str
    index: int
    start_char: int
    end_char: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_document(self, source_id: str | None = None) -> Document:
        """Convert chunk to Document for storage."""
        metadata = {
            **self.metadata,
            "chunk_index": self.index,
            "start_char": self.start_char,
            "end_char": self.end_char,
        }
        if source_id:
            metadata["source_id"] = source_id

        return Document(
            id=str(uuid.uuid4()),
            content=self.content,
            metadata=metadata,
        )


class TextChunker(ABC):
    """Abstract text chunker interface."""

    @abstractmethod
    def chunk(self, text: str, metadata: dict[str, Any] | None = None) -> list[Chunk]:
        """
        Split text into chunks.

        Args:
            text: Text to chunk.
            metadata: Metadata to attach to each chunk.

        Returns:
            List of chunks.
        """
        pass


class FixedSizeChunker(TextChunker):
    """
    Fixed-size chunking with overlap.

    Simple and predictable, good for most use cases.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        """
        Initialize fixed-size chunker.

        Args:
            chunk_size: Target size of each chunk in characters.
            chunk_overlap: Overlap between chunks in characters.
        """
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, text: str, metadata: dict[str, Any] | None = None) -> list[Chunk]:
        """Split text into fixed-size chunks."""
        if not text:
            return []

        metadata = metadata or {}
        chunks = []
        start = 0
        index = 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))

            chunk_text = text[start:end]
            chunks.append(Chunk(
                content=chunk_text,
                index=index,
                start_char=start,
                end_char=end,
                metadata=metadata.copy(),
            ))

            # Move start position, accounting for overlap
            start = end - self.chunk_overlap
            index += 1

            # Avoid infinite loop at end
            if start >= len(text) - self.chunk_overlap:
                break

        return chunks


class SemanticChunker(TextChunker):
    """
    Semantic chunking that respects sentence boundaries.

    Tries to keep related content together while staying
    within size limits.
    """

    # Sentence-ending patterns
    SENTENCE_ENDINGS = re.compile(r'(?<=[.!?])\s+')

    # Paragraph patterns
    PARAGRAPH_BREAK = re.compile(r'\n\n+')

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        min_chunk_size: int = 100,
    ):
        """
        Initialize semantic chunker.

        Args:
            chunk_size: Target chunk size in characters.
            chunk_overlap: Overlap between chunks.
            min_chunk_size: Minimum chunk size (avoid tiny chunks).
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

    def _split_into_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        # First split by paragraphs to preserve structure
        paragraphs = self.PARAGRAPH_BREAK.split(text)

        sentences = []
        for para in paragraphs:
            # Split paragraph into sentences
            para_sentences = self.SENTENCE_ENDINGS.split(para)
            for sent in para_sentences:
                sent = sent.strip()
                if sent:
                    sentences.append(sent)

        return sentences

    def chunk(self, text: str, metadata: dict[str, Any] | None = None) -> list[Chunk]:
        """Split text into semantic chunks."""
        if not text:
            return []

        metadata = metadata or {}
        sentences = self._split_into_sentences(text)

        if not sentences:
            return []

        chunks = []
        current_chunk: list[str] = []
        current_size = 0
        chunk_start = 0
        current_pos = 0
        index = 0

        for sentence in sentences:
            sentence_len = len(sentence)

            # If single sentence is too long, split it
            if sentence_len > self.chunk_size:
                # Flush current chunk first
                if current_chunk:
                    chunk_text = " ".join(current_chunk)
                    chunks.append(Chunk(
                        content=chunk_text,
                        index=index,
                        start_char=chunk_start,
                        end_char=current_pos,
                        metadata=metadata.copy(),
                    ))
                    index += 1
                    current_chunk = []
                    current_size = 0

                # Split long sentence into smaller pieces
                for i in range(0, sentence_len, self.chunk_size - self.chunk_overlap):
                    piece = sentence[i:i + self.chunk_size]
                    chunks.append(Chunk(
                        content=piece,
                        index=index,
                        start_char=current_pos + i,
                        end_char=current_pos + i + len(piece),
                        metadata=metadata.copy(),
                    ))
                    index += 1

                chunk_start = current_pos + sentence_len
                current_pos += sentence_len + 1
                continue

            # Check if adding sentence exceeds chunk size
            if current_size + sentence_len + 1 > self.chunk_size:
                # Save current chunk if it meets minimum size
                if current_size >= self.min_chunk_size:
                    chunk_text = " ".join(current_chunk)
                    chunks.append(Chunk(
                        content=chunk_text,
                        index=index,
                        start_char=chunk_start,
                        end_char=current_pos,
                        metadata=metadata.copy(),
                    ))
                    index += 1

                    # Start new chunk with overlap
                    # Keep last few sentences for context
                    overlap_size = 0
                    overlap_sentences = []
                    for s in reversed(current_chunk):
                        if overlap_size + len(s) > self.chunk_overlap:
                            break
                        overlap_sentences.insert(0, s)
                        overlap_size += len(s) + 1

                    current_chunk = overlap_sentences
                    current_size = overlap_size
                    chunk_start = current_pos - overlap_size

            current_chunk.append(sentence)
            current_size += sentence_len + 1
            current_pos += sentence_len + 1

        # Don't forget the last chunk
        if current_chunk and current_size >= self.min_chunk_size:
            chunk_text = " ".join(current_chunk)
            chunks.append(Chunk(
                content=chunk_text,
                index=index,
                start_char=chunk_start,
                end_char=len(text),
                metadata=metadata.copy(),
            ))

        return chunks


class RecursiveChunker(TextChunker):
    """
    Recursive chunking for structured documents.

    Tries to split on natural boundaries in order:
    1. Double newlines (paragraphs)
    2. Single newlines
    3. Sentences
    4. Words
    5. Characters
    """

    SEPARATORS = [
        "\n\n",  # Paragraphs
        "\n",    # Lines
        ". ",    # Sentences
        ", ",    # Clauses
        " ",     # Words
        "",      # Characters
    ]

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _split_text(self, text: str, separators: list[str]) -> list[str]:
        """Recursively split text using separators."""
        if not text:
            return []

        separator = separators[0]
        remaining_separators = separators[1:]

        if separator == "":
            # Base case: split into characters
            return list(text)

        splits = text.split(separator)

        # If we have remaining separators, try to split large pieces further
        if remaining_separators:
            result = []
            for split in splits:
                if len(split) > self.chunk_size:
                    result.extend(self._split_text(split, remaining_separators))
                else:
                    result.append(split)
            return result

        return splits

    def chunk(self, text: str, metadata: dict[str, Any] | None = None) -> list[Chunk]:
        """Split text recursively."""
        if not text:
            return []

        metadata = metadata or {}
        splits = self._split_text(text, self.SEPARATORS)

        # Merge splits into chunks
        chunks = []
        current_chunk: list[str] = []
        current_size = 0
        chunk_start = 0
        current_pos = 0
        index = 0

        for split in splits:
            split_len = len(split)

            if current_size + split_len > self.chunk_size and current_chunk:
                chunk_text = "".join(current_chunk)
                chunks.append(Chunk(
                    content=chunk_text,
                    index=index,
                    start_char=chunk_start,
                    end_char=current_pos,
                    metadata=metadata.copy(),
                ))
                index += 1

                # Handle overlap
                overlap_text = chunk_text[-self.chunk_overlap:] if len(chunk_text) > self.chunk_overlap else ""
                current_chunk = [overlap_text] if overlap_text else []
                current_size = len(overlap_text)
                chunk_start = current_pos - len(overlap_text)

            current_chunk.append(split)
            current_size += split_len
            current_pos += split_len

        # Last chunk
        if current_chunk:
            chunk_text = "".join(current_chunk)
            chunks.append(Chunk(
                content=chunk_text,
                index=index,
                start_char=chunk_start,
                end_char=len(text),
                metadata=metadata.copy(),
            ))

        return chunks


def create_chunker(
    strategy: str = "semantic",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    **kwargs,
) -> TextChunker:
    """
    Factory function to create chunker.

    Args:
        strategy: "fixed", "semantic", or "recursive"
        chunk_size: Target chunk size.
        chunk_overlap: Overlap between chunks.
        **kwargs: Additional chunker arguments.

    Returns:
        Configured chunker.
    """
    chunkers = {
        "fixed": FixedSizeChunker,
        "semantic": SemanticChunker,
        "recursive": RecursiveChunker,
    }

    if strategy not in chunkers:
        raise ValueError(f"Unknown chunking strategy: {strategy}")

    return chunkers[strategy](
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        **kwargs,
    )
