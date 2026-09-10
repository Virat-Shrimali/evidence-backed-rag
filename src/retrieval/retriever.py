"""Independent retriever abstractions for dense, sparse, hybrid, and reranked search."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Literal

from src.config import RAGConfig, settings
from src.retrieval.hybrid import hybrid_retrieve
from src.retrieval.models import RetrievedChunk
from src.retrieval.rerank import CrossEncoderReranker

if TYPE_CHECKING:
    from src.index.bm25_index import BM25Index
    from src.index.embed import DenseIndex


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


class RerankedRetriever(BaseRetriever):
    """Two-stage retriever applying a neural cross-encoder reranker to any base retriever."""

    def __init__(
        self,
        base_retriever: BaseRetriever,
        reranker: CrossEncoderReranker | None = None,
        candidate_top_k: int = 20,
        final_top_k: int = 5,
    ):
        self.base_retriever = base_retriever
        self.reranker = reranker or CrossEncoderReranker()
        self.candidate_top_k = candidate_top_k
        self.final_top_k = final_top_k

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        k_final = top_k or self.final_top_k

        # Stage 1: Fetch candidate pool from base retriever (BM25, Dense, or Hybrid)
        candidates = self.base_retriever.retrieve(query, top_k=self.candidate_top_k)
        if not candidates:
            return []

        # Stage 2: Cross-encoder reranks candidates down to final_top_k
        return self.reranker.rerank(query=query, candidates=candidates, top_k=k_final)


def create_retriever(
    strategy: Literal["dense_only", "bm25_only", "hybrid", "hybrid_rerank"]
    | str
    | None = None,
    dense_index: Any | None = None,
    bm25_index: Any | None = None,
    reranker: CrossEncoderReranker | None = None,
    config: RAGConfig | None = None,
) -> BaseRetriever:
    """Factory creating an independently callable retriever based on strategy configuration."""
    cfg = config or settings
    active_strategy = strategy or cfg.retrieval_strategy

    if dense_index is None:
        from src.index.embed import DenseIndex

        d_index = DenseIndex(
            persist_dir=cfg.chroma_persist_dir,
            model_name=cfg.embedding_model_name,
        )
    else:
        d_index = dense_index

    if bm25_index is None:
        from src.index.bm25_index import BM25Index

        b_index = BM25Index()
        if b_index.count() == 0 and cfg.raw_data_dir.exists():
            from src.ingest.chunk import chunk_document
            from src.ingest.parse import parse_directory

            docs = parse_directory(cfg.raw_data_dir)
            if docs:
                corpus_chunks = []
                for doc in docs:
                    corpus_chunks.extend(
                        chunk_document(doc, strategy=cfg.active_chunking_strategy)
                    )
                if corpus_chunks:
                    b_index.index_chunks(corpus_chunks)
                    col = d_index.get_collection(collection_name="rag_chunks")
                    if col.count() == 0:
                        d_index.index_chunks(corpus_chunks)
    else:
        b_index = bm25_index

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
    elif active_strategy in ("hybrid_rerank", "hybrid_reranker"):
        hybrid_base = HybridRetriever(
            dense_retriever=dense_retriever,
            bm25_retriever=bm25_retriever,
            rrf_k=cfg.rrf_k,
            dense_top_k=cfg.dense_top_k,
            sparse_top_k=cfg.sparse_top_k,
            final_top_k=cfg.candidate_top_k,
        )
        return RerankedRetriever(
            base_retriever=hybrid_base,
            reranker=reranker
            or CrossEncoderReranker(model_name=cfg.reranker_model_name),
            candidate_top_k=cfg.candidate_top_k,
            final_top_k=cfg.final_top_k,
        )
    else:
        raise ValueError(f"Unknown retrieval strategy: {active_strategy}")
