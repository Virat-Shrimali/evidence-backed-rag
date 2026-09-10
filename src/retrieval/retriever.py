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


# Module-level singletons for index, model, and chunk caching
_shared_bm25_index: BM25Index | None = None
_shared_dense_index: DenseIndex | None = None
_shared_reranker: CrossEncoderReranker | None = None
_shared_corpus_chunks: list[Any] | None = None


def clear_retriever_cache() -> None:
    """Reset all module-level retriever and index singletons."""
    global _shared_bm25_index, _shared_dense_index, _shared_reranker, _shared_corpus_chunks
    _shared_bm25_index = None
    _shared_dense_index = None
    _shared_reranker = None
    _shared_corpus_chunks = None


def _get_corpus_chunks(cfg: RAGConfig) -> list[Any]:
    """Lazily parse and chunk corpus documents once."""
    global _shared_corpus_chunks
    if _shared_corpus_chunks is not None:
        return _shared_corpus_chunks

    chunks: list[Any] = []
    if cfg.raw_data_dir.exists():
        from src.ingest.chunk import chunk_document
        from src.ingest.parse import parse_directory

        docs = parse_directory(cfg.raw_data_dir)
        for doc in docs:
            chunks.extend(chunk_document(doc, strategy=cfg.active_chunking_strategy))
    _shared_corpus_chunks = chunks
    return _shared_corpus_chunks


def _get_bm25_index(
    bm25_index: Any | None, cfg: RAGConfig, auto_index: bool = True
) -> BM25Index:
    """Resolve or lazily initialize shared BM25Index without loading dense/reranker models."""
    if bm25_index is not None:
        return bm25_index

    global _shared_bm25_index
    if _shared_bm25_index is None:
        from src.index.bm25_index import BM25Index

        _shared_bm25_index = BM25Index()

    if auto_index and _shared_bm25_index.count() == 0 and cfg.raw_data_dir.exists():
        chunks = _get_corpus_chunks(cfg)
        if chunks:
            _shared_bm25_index.index_chunks(chunks)

    return _shared_bm25_index


def _get_dense_index(
    dense_index: Any | None, cfg: RAGConfig, auto_index: bool = True
) -> DenseIndex:
    """Resolve or lazily initialize shared DenseIndex without loading BM25 or reranker models."""
    if dense_index is not None:
        return dense_index

    global _shared_dense_index
    if _shared_dense_index is None:
        from src.index.embed import DenseIndex

        _shared_dense_index = DenseIndex(
            persist_dir=cfg.chroma_persist_dir,
            model_name=cfg.embedding_model_name,
        )

    if auto_index and cfg.raw_data_dir.exists():
        col = _shared_dense_index.get_collection(collection_name="rag_chunks")
        if col.count() == 0:
            chunks = _get_corpus_chunks(cfg)
            if chunks:
                _shared_dense_index.index_chunks(chunks)

    return _shared_dense_index


def _get_reranker(
    reranker: CrossEncoderReranker | None, cfg: RAGConfig
) -> CrossEncoderReranker:
    """Resolve or lazily initialize shared CrossEncoderReranker."""
    if reranker is not None:
        return reranker

    global _shared_reranker
    if _shared_reranker is None:
        _shared_reranker = CrossEncoderReranker(model_name=cfg.reranker_model_name)
    return _shared_reranker


def create_retriever(
    strategy: Literal["dense_only", "bm25_only", "hybrid", "hybrid_rerank"]
    | str
    | None = None,
    dense_index: Any | None = None,
    bm25_index: Any | None = None,
    reranker: CrossEncoderReranker | None = None,
    config: RAGConfig | None = None,
) -> BaseRetriever:
    """Factory creating an independently callable retriever based on strategy configuration.

    Only initializes the models and indexes required for the selected strategy:
    - 'bm25_only': Only loads BM25Index. Zero PyTorch, zero SentenceTransformers, zero CrossEncoder.
    - 'dense_only': Only loads DenseIndex and embedding model.
    - 'hybrid': Loads BM25Index and DenseIndex.
    - 'hybrid_rerank': Loads BM25Index, DenseIndex, and CrossEncoderReranker.
    """
    cfg = config or settings
    active_strategy = strategy or cfg.retrieval_strategy

    if active_strategy == "bm25_only":
        b_index = _get_bm25_index(bm25_index, cfg)
        return BM25Retriever(bm25_index=b_index, top_k=cfg.sparse_top_k)

    elif active_strategy == "dense_only":
        d_index = _get_dense_index(dense_index, cfg)
        return DenseRetriever(dense_index=d_index, top_k=cfg.dense_top_k)

    elif active_strategy == "hybrid":
        b_index = _get_bm25_index(bm25_index, cfg)
        d_index = _get_dense_index(dense_index, cfg)
        dense_retriever = DenseRetriever(dense_index=d_index, top_k=cfg.dense_top_k)
        bm25_retriever = BM25Retriever(bm25_index=b_index, top_k=cfg.sparse_top_k)
        return HybridRetriever(
            dense_retriever=dense_retriever,
            bm25_retriever=bm25_retriever,
            rrf_k=cfg.rrf_k,
            dense_top_k=cfg.dense_top_k,
            sparse_top_k=cfg.sparse_top_k,
            final_top_k=cfg.final_top_k,
        )

    elif active_strategy in ("hybrid_rerank", "hybrid_reranker"):
        b_index = _get_bm25_index(bm25_index, cfg)
        d_index = _get_dense_index(dense_index, cfg)
        dense_retriever = DenseRetriever(dense_index=d_index, top_k=cfg.dense_top_k)
        bm25_retriever = BM25Retriever(bm25_index=b_index, top_k=cfg.sparse_top_k)
        hybrid_base = HybridRetriever(
            dense_retriever=dense_retriever,
            bm25_retriever=bm25_retriever,
            rrf_k=cfg.rrf_k,
            dense_top_k=cfg.dense_top_k,
            sparse_top_k=cfg.sparse_top_k,
            final_top_k=cfg.candidate_top_k,
        )
        rerank = _get_reranker(reranker, cfg)
        return RerankedRetriever(
            base_retriever=hybrid_base,
            reranker=rerank,
            candidate_top_k=cfg.candidate_top_k,
            final_top_k=cfg.final_top_k,
        )
    else:
        raise ValueError(f"Unknown retrieval strategy: {active_strategy}")
