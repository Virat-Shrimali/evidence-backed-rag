"""Independent retriever abstractions for dense, sparse, and hybrid search."""

from abc import ABC, abstractmethod
from typing import Literal

from src.config import RAGConfig, settings
from src.index.bm25_index import BM25Index
from src.index.embed import DenseIndex
from src.retrieval.hybrid import hybrid_retrieve
from src.retrieval.models import RetrievedChunk


class BaseRetriever(ABC):
    """Abstract base class for chunk retrieval strategies."""

    @abstractmethod
    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        """Retrieve ranked chunks for a query string."""
        pass


class DenseRetriever(BaseRetriever):
    """Dense vector retriever using ChromaDB and SentenceTransformers."""

    def __init__(
        self,
        dense_index: DenseIndex,
        top_k: int = 5,
        collection_name: str = "rag_chunks",
    ):
        self.dense_index = dense_index
        self.top_k = top_k
        self.collection_name = collection_name

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        k = top_k or self.top_k
        return self.dense_index.search(
            query=query,
            top_k=k,
            collection_name=self.collection_name,
        )


class BM25Retriever(BaseRetriever):
    """Sparse keyword retriever using Okapi BM25."""

    def __init__(self, bm25_index: BM25Index, top_k: int = 5):
        self.bm25_index = bm25_index
        self.top_k = top_k

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        k = top_k or self.top_k
        return self.bm25_index.search(query=query, top_k=k)


class HybridRetriever(BaseRetriever):
    """Hybrid retriever combining dense and BM25 retrievers via Reciprocal Rank Fusion."""

    def __init__(
        self,
        dense_retriever: DenseRetriever,
        bm25_retriever: BM25Retriever,
        rrf_k: int = 60,
        dense_top_k: int = 10,
        sparse_top_k: int = 10,
        final_top_k: int = 5,
    ):
        self.dense_retriever = dense_retriever
        self.bm25_retriever = bm25_retriever
        self.rrf_k = rrf_k
        self.dense_top_k = dense_top_k
        self.sparse_top_k = sparse_top_k
        self.final_top_k = final_top_k

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        k_final = top_k or self.final_top_k

        # Fetch candidates independently from dense and sparse retrievers
        dense_candidates = self.dense_retriever.retrieve(query, top_k=self.dense_top_k)
        sparse_candidates = self.bm25_retriever.retrieve(query, top_k=self.sparse_top_k)

        # Fuse via Reciprocal Rank Fusion
        return hybrid_retrieve(
            dense_results=dense_candidates,
            sparse_results=sparse_candidates,
            k=self.rrf_k,
            top_k=k_final,
        )


def create_retriever(
    strategy: Literal["dense_only", "bm25_only", "hybrid"] | str | None = None,
    dense_index: DenseIndex | None = None,
    bm25_index: BM25Index | None = None,
    config: RAGConfig | None = None,
) -> BaseRetriever:
    """Factory creating an independently callable retriever based on strategy configuration."""
    cfg = config or settings
    active_strategy = strategy or cfg.retrieval_strategy

    d_index = dense_index or DenseIndex(
        persist_dir=cfg.chroma_persist_dir,
        model_name=cfg.embedding_model_name,
    )
    b_index = bm25_index or BM25Index()

    dense_retriever = DenseRetriever(dense_index=d_index, top_k=cfg.dense_top_k)
    bm25_retriever = BM25Retriever(bm25_index=b_index, top_k=cfg.sparse_top_k)

    if active_strategy == "dense_only":
        return dense_retriever
    elif active_strategy == "bm25_only":
        return bm25_retriever
    elif active_strategy == "hybrid":
        return HybridRetriever(
            dense_retriever=dense_retriever,
            bm25_retriever=bm25_retriever,
            rrf_k=cfg.rrf_k,
            dense_top_k=cfg.dense_top_k,
            sparse_top_k=cfg.sparse_top_k,
            final_top_k=cfg.final_top_k,
        )
    else:
        raise ValueError(f"Unknown retrieval strategy: {active_strategy}")
