"""
Retrieval strategies for RAG pipeline.

Supports:
- Vector similarity search (with embeddings)
- BM25 keyword search (without embeddings)
- Hybrid search (combining both)
"""
import math
import re
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from app.rag.embeddings import EmbeddingProvider
from app.rag.vector_store import Document, SearchResult, VectorStore


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


class Reranker(ABC):
    """Abstract reranker interface for two-stage retrieval."""

    @abstractmethod
    async def rerank(
        self,
        query: str,
        results: list[SearchResult],
        limit: int | None = None,
    ) -> list[SearchResult]:
        """
        Rerank retrieval results.

        Args:
            query: Original search query.
            results: Initial retrieval results.
            limit: Optional limit on returned results.

        Returns:
            Reranked results sorted by relevance.
        """
        pass


class CrossEncoderReranker(Reranker):
    """
    Reranker using cross-encoder style scoring.

    Instead of separate query and document embeddings, this computes
    a relevance score by embedding the concatenation of query and document.
    This is more accurate but more expensive than bi-encoder retrieval.
    """

    def __init__(
        self,
        embedder: EmbeddingProvider,
        score_threshold: float = 0.0,
    ):
        """
        Initialize cross-encoder reranker.

        Args:
            embedder: Embedding provider for scoring.
            score_threshold: Minimum score to include.
        """
        self.embedder = embedder
        self.score_threshold = score_threshold

    async def rerank(
        self,
        query: str,
        results: list[SearchResult],
        limit: int | None = None,
    ) -> list[SearchResult]:
        """Rerank using cross-encoder style scoring."""
        if not results:
            return []

        # Generate query embedding
        query_embedding = await self.embedder.embed_query(query)

        # Score each document by embedding similarity to query
        scored_results: list[tuple[SearchResult, float]] = []

        for result in results:
            # Get document embedding - recompute for accuracy
            doc_embedding = await self.embedder.embed_query(result.document.content)

            # Compute cosine similarity
            score = self._cosine_similarity(query_embedding, doc_embedding)

            if score >= self.score_threshold:
                scored_results.append((result, score))

        # Sort by new score
        scored_results.sort(key=lambda x: x[1], reverse=True)

        # Apply limit
        if limit:
            scored_results = scored_results[:limit]

        # Create new results with updated scores and ranks
        return [
            SearchResult(
                document=res.document,
                score=score,
                rank=i + 1,
            )
            for i, (res, score) in enumerate(scored_results)
        ]

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)


class WeightedScoreReranker(Reranker):
    """
    Reranker that combines original score with additional signals.

    Useful for incorporating metadata signals like recency, popularity, etc.
    """

    def __init__(
        self,
        original_weight: float = 0.7,
        recency_weight: float = 0.15,
        length_weight: float = 0.15,
        recency_field: str = "created_at",
        preferred_length: int = 500,
    ):
        """
        Initialize weighted reranker.

        Args:
            original_weight: Weight for original retrieval score.
            recency_weight: Weight for recency signal.
            length_weight: Weight for content length signal.
            recency_field: Metadata field for timestamp.
            preferred_length: Optimal content length.
        """
        self.original_weight = original_weight
        self.recency_weight = recency_weight
        self.length_weight = length_weight
        self.recency_field = recency_field
        self.preferred_length = preferred_length

    async def rerank(
        self,
        query: str,
        results: list[SearchResult],
        limit: int | None = None,
    ) -> list[SearchResult]:
        """Rerank using weighted combination of signals."""
        if not results:
            return []

        scored_results: list[tuple[SearchResult, float]] = []

        # Find max timestamp for normalization
        max_timestamp = 0.0
        for result in results:
            ts = result.document.metadata.get(self.recency_field, 0)
            if isinstance(ts, (int, float)):
                max_timestamp = max(max_timestamp, ts)

        for result in results:
            # Original score (normalized to 0-1)
            original_score = max(0.0, min(1.0, result.score))

            # Recency score
            recency_score = 0.5  # Default if no timestamp
            ts = result.document.metadata.get(self.recency_field, 0)
            if isinstance(ts, (int, float)) and max_timestamp > 0:
                recency_score = ts / max_timestamp

            # Length score - prefer documents near preferred length
            content_len = len(result.document.content)
            length_diff = abs(content_len - self.preferred_length)
            length_score = max(0.0, 1.0 - (length_diff / self.preferred_length))

            # Combined score
            combined_score = (
                self.original_weight * original_score
                + self.recency_weight * recency_score
                + self.length_weight * length_score
            )

            scored_results.append((result, combined_score))

        # Sort by combined score
        scored_results.sort(key=lambda x: x[1], reverse=True)

        if limit:
            scored_results = scored_results[:limit]

        return [
            SearchResult(
                document=res.document,
                score=score,
                rank=i + 1,
            )
            for i, (res, score) in enumerate(scored_results)
        ]


class RetrievalPipeline:
    """
    Two-stage retrieval pipeline: retrieve then rerank.

    Combines a retriever for initial candidate generation with
    a reranker for more accurate final ranking.
    """

    def __init__(
        self,
        retriever: Retriever,
        reranker: Reranker | None = None,
        initial_limit_multiplier: int = 3,
    ):
        """
        Initialize retrieval pipeline.

        Args:
            retriever: First stage retriever.
            reranker: Optional second stage reranker.
            initial_limit_multiplier: Factor to multiply limit for initial retrieval.
        """
        self.retriever = retriever
        self.reranker = reranker
        self.initial_limit_multiplier = initial_limit_multiplier

    async def retrieve(
        self,
        query: str,
        limit: int = 10,
        filter: dict[str, Any] | None = None,
    ) -> RetrievalResult:
        """
        Execute two-stage retrieval.

        Args:
            query: Search query.
            limit: Final number of results.
            filter: Metadata filter.

        Returns:
            RetrievalResult with ranked documents.
        """
        # First stage: initial retrieval
        initial_limit = limit * self.initial_limit_multiplier if self.reranker else limit
        initial_result = await self.retriever.retrieve(
            query=query,
            limit=initial_limit,
            filter=filter,
        )

        if not self.reranker:
            return initial_result

        # Second stage: reranking
        reranked_docs = await self.reranker.rerank(
            query=query,
            results=initial_result.documents,
            limit=limit,
        )

        return RetrievalResult(
            documents=reranked_docs,
            query=query,
            strategy=f"{initial_result.strategy}+rerank",
            metadata={
                **initial_result.metadata,
                "reranked": True,
                "initial_candidates": len(initial_result.documents),
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
