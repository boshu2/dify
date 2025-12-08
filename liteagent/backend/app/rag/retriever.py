"""
Retrieval strategies for RAG pipeline.

Supports:
- Vector similarity search (with embeddings)
- BM25 keyword search (without embeddings)
- Hybrid search (combining both)
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
import math
import re
from collections import Counter

from app.rag.vector_store import VectorStore, Document, SearchResult
from app.rag.embeddings import EmbeddingProvider


@dataclass
class RetrievalResult:
    """Result from retrieval operation."""
    documents: list[SearchResult]
    query: str
    strategy: str
    metadata: dict[str, Any] = field(default_factory=dict)


class Retriever(ABC):
    """Abstract retriever interface."""

    @abstractmethod
    async def retrieve(
        self,
        query: str,
        limit: int = 10,
        filter: dict[str, Any] | None = None,
    ) -> RetrievalResult:
        """
        Retrieve relevant documents for a query.

        Args:
            query: Search query.
            limit: Max documents to return.
            filter: Metadata filter.

        Returns:
            RetrievalResult with ranked documents.
        """
        pass


class VectorRetriever(Retriever):
    """
    Retriever using vector similarity search.
    Requires embeddings.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        embedder: EmbeddingProvider,
    ):
        self.vector_store = vector_store
        self.embedder = embedder

    async def retrieve(
        self,
        query: str,
        limit: int = 10,
        filter: dict[str, Any] | None = None,
    ) -> RetrievalResult:
        # Embed the query
        query_embedding = await self.embedder.embed_query(query)

        # Search vector store
        results = await self.vector_store.search(
            query_embedding=query_embedding,
            limit=limit,
            filter=filter,
        )

        return RetrievalResult(
            documents=results,
            query=query,
            strategy="vector",
            metadata={"model": self.embedder.model_name},
        )


class BM25Retriever(Retriever):
    """
    BM25 keyword-based retriever.
    No embeddings required - uses term frequency.

    BM25 formula:
    score(D,Q) = Σ IDF(qi) * (f(qi,D) * (k1 + 1)) / (f(qi,D) + k1 * (1 - b + b * |D|/avgdl))

    Where:
    - f(qi,D) = term frequency of qi in document D
    - |D| = document length
    - avgdl = average document length
    - k1, b = tuning parameters
    """

    def __init__(
        self,
        documents: list[Document] | None = None,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        """
        Initialize BM25 retriever.

        Args:
            documents: Initial documents to index.
            k1: Term frequency saturation parameter.
            b: Document length normalization parameter.
        """
        self.k1 = k1
        self.b = b

        self._documents: list[Document] = []
        self._doc_freqs: Counter = Counter()  # Term -> doc count
        self._doc_lengths: list[int] = []
        self._avgdl: float = 0.0
        self._tokenized_docs: list[list[str]] = []

        if documents:
            self.add_documents(documents)

    def _tokenize(self, text: str) -> list[str]:
        """Simple tokenization - lowercase and split on non-alphanumeric."""
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens

    def add_documents(self, documents: list[Document]) -> None:
        """Add documents to the BM25 index."""
        for doc in documents:
            tokens = self._tokenize(doc.content)
            self._documents.append(doc)
            self._tokenized_docs.append(tokens)
            self._doc_lengths.append(len(tokens))

            # Update document frequencies
            unique_tokens = set(tokens)
            for token in unique_tokens:
                self._doc_freqs[token] += 1

        # Update average document length
        if self._doc_lengths:
            self._avgdl = sum(self._doc_lengths) / len(self._doc_lengths)

    def _idf(self, term: str) -> float:
        """Calculate inverse document frequency."""
        n = len(self._documents)
        df = self._doc_freqs.get(term, 0)

        if df == 0:
            return 0.0

        # Standard BM25 IDF formula
        return math.log((n - df + 0.5) / (df + 0.5) + 1)

    def _score(self, query_tokens: list[str], doc_idx: int) -> float:
        """Calculate BM25 score for a document."""
        doc_tokens = self._tokenized_docs[doc_idx]
        doc_len = self._doc_lengths[doc_idx]

        # Count term frequencies in document
        term_freqs = Counter(doc_tokens)

        score = 0.0
        for term in query_tokens:
            if term not in term_freqs:
                continue

            tf = term_freqs[term]
            idf = self._idf(term)

            # BM25 scoring formula
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self._avgdl)

            score += idf * (numerator / denominator)

        return score

    async def retrieve(
        self,
        query: str,
        limit: int = 10,
        filter: dict[str, Any] | None = None,
    ) -> RetrievalResult:
        """Retrieve using BM25 scoring."""
        query_tokens = self._tokenize(query)

        if not query_tokens or not self._documents:
            return RetrievalResult(
                documents=[],
                query=query,
                strategy="bm25",
            )

        # Score all documents
        scores = []
        for idx, doc in enumerate(self._documents):
            # Apply filter
            if filter:
                skip = False
                for key, value in filter.items():
                    if doc.metadata.get(key) != value:
                        skip = True
                        break
                if skip:
                    continue

            score = self._score(query_tokens, idx)
            if score > 0:
                scores.append((idx, score))

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)

        # Build results
        results = []
        for rank, (idx, score) in enumerate(scores[:limit]):
            results.append(SearchResult(
                document=self._documents[idx],
                score=score,
                rank=rank + 1,
            ))

        return RetrievalResult(
            documents=results,
            query=query,
            strategy="bm25",
            metadata={"k1": self.k1, "b": self.b},
        )


class HybridRetriever(Retriever):
    """
    Hybrid retriever combining vector search and BM25.

    Uses Reciprocal Rank Fusion (RRF) to combine results:
    RRF(d) = Σ 1 / (k + rank(d))

    This gives good results without needing to normalize scores
    between different retrieval methods.
    """

    def __init__(
        self,
        vector_retriever: VectorRetriever | None = None,
        bm25_retriever: BM25Retriever | None = None,
        vector_weight: float = 0.5,
        rrf_k: int = 60,
    ):
        """
        Initialize hybrid retriever.

        Args:
            vector_retriever: Vector similarity retriever.
            bm25_retriever: BM25 keyword retriever.
            vector_weight: Weight for vector results (0-1).
            rrf_k: RRF constant (higher = less emphasis on top ranks).
        """
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.vector_weight = vector_weight
        self.bm25_weight = 1.0 - vector_weight
        self.rrf_k = rrf_k

    async def retrieve(
        self,
        query: str,
        limit: int = 10,
        filter: dict[str, Any] | None = None,
    ) -> RetrievalResult:
        """Retrieve using hybrid approach."""
        # Get results from both retrievers
        vector_results: list[SearchResult] = []
        bm25_results: list[SearchResult] = []

        if self.vector_retriever and self.vector_weight > 0:
            result = await self.vector_retriever.retrieve(
                query, limit=limit * 2, filter=filter
            )
            vector_results = result.documents

        if self.bm25_retriever and self.bm25_weight > 0:
            result = await self.bm25_retriever.retrieve(
                query, limit=limit * 2, filter=filter
            )
            bm25_results = result.documents

        # Combine using RRF
        doc_scores: dict[str, tuple[Document, float]] = {}

        # Score vector results
        for result in vector_results:
            doc_id = result.document.id
            rrf_score = self.vector_weight / (self.rrf_k + result.rank)

            if doc_id in doc_scores:
                doc_scores[doc_id] = (
                    result.document,
                    doc_scores[doc_id][1] + rrf_score
                )
            else:
                doc_scores[doc_id] = (result.document, rrf_score)

        # Score BM25 results
        for result in bm25_results:
            doc_id = result.document.id
            rrf_score = self.bm25_weight / (self.rrf_k + result.rank)

            if doc_id in doc_scores:
                doc_scores[doc_id] = (
                    result.document,
                    doc_scores[doc_id][1] + rrf_score
                )
            else:
                doc_scores[doc_id] = (result.document, rrf_score)

        # Sort by combined score
        sorted_docs = sorted(
            doc_scores.values(),
            key=lambda x: x[1],
            reverse=True,
        )

        # Build final results
        results = [
            SearchResult(
                document=doc,
                score=score,
                rank=i + 1,
            )
            for i, (doc, score) in enumerate(sorted_docs[:limit])
        ]

        return RetrievalResult(
            documents=results,
            query=query,
            strategy="hybrid",
            metadata={
                "vector_weight": self.vector_weight,
                "bm25_weight": self.bm25_weight,
                "rrf_k": self.rrf_k,
            },
        )


def create_retriever(
    strategy: str = "hybrid",
    vector_store: VectorStore | None = None,
    embedder: EmbeddingProvider | None = None,
    documents: list[Document] | None = None,
    **kwargs,
) -> Retriever:
    """
    Factory function to create retriever.

    Args:
        strategy: "vector", "bm25", or "hybrid"
        vector_store: Vector store for similarity search.
        embedder: Embedding provider.
        documents: Documents for BM25 index.
        **kwargs: Additional retriever arguments.

    Returns:
        Configured retriever.
    """
    if strategy == "vector":
        if not vector_store or not embedder:
            raise ValueError("Vector retriever requires vector_store and embedder")
        return VectorRetriever(vector_store, embedder)

    elif strategy == "bm25":
        return BM25Retriever(documents=documents, **kwargs)

    elif strategy == "hybrid":
        vector_ret = None
        bm25_ret = None

        if vector_store and embedder:
            vector_ret = VectorRetriever(vector_store, embedder)

        if documents:
            bm25_ret = BM25Retriever(documents=documents)

        return HybridRetriever(
            vector_retriever=vector_ret,
            bm25_retriever=bm25_ret,
            **kwargs,
        )

    else:
        raise ValueError(f"Unknown retrieval strategy: {strategy}")
