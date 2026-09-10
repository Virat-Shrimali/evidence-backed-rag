"""Comprehensive unit and integration tests for dense, BM25, and hybrid retrieval."""

from pathlib import Path

import pytest

from evaluation.retrieval_metrics import (
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
)
from src.index.bm25_index import BM25Index, tokenize_for_bm25
from src.index.embed import DenseIndex
from src.ingest.chunk import Chunk
from src.retrieval.hybrid import hybrid_retrieve, reciprocal_rank_fusion
from src.retrieval.models import RetrievedChunk
from src.retrieval.retriever import (
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    create_retriever,
)


@pytest.fixture
def retrieval_test_chunks() -> list[Chunk]:
    """Sample chunk corpus for retrieval testing."""
    return [
        Chunk(
            id="doc1#stratA#c0000",
            document_id="doc1",
            content="Apple Inc. reported quarterly revenue of 90 billion dollars driven by iPhone sales.",
            token_count=16,
            chunk_index=0,
            page_numbers=[1],
            metadata={"category": "earnings", "source": "apple_10q.pdf"},
        ),
        Chunk(
            id="doc1#stratA#c0001",
            document_id="doc1",
            content="Operating expenses in research and development rose by 12 percent year over year.",
            token_count=15,
            chunk_index=1,
            page_numbers=[1, 2],
            metadata={"category": "expenses", "source": "apple_10q.pdf"},
        ),
        Chunk(
            id="doc2#stratA#c0000",
            document_id="doc2",
            content="Microsoft Cloud revenue reached 35 billion dollars with substantial Azure AI growth.",
            token_count=14,
            chunk_index=0,
            page_numbers=[1],
            metadata={"category": "cloud", "source": "msft_10q.pdf"},
        ),
        Chunk(
            id="doc2#stratA#c0001",
            document_id="doc2",
            content="Primary business risk factors include cybersecurity incidents and regulatory compliance.",
            token_count=13,
            chunk_index=1,
            page_numbers=[2],
            metadata={"category": "risks", "source": "msft_10q.pdf"},
        ),
    ]


# ---------------------------------------------------------------------------
# 1. BM25 Sparse Retrieval Tests
# ---------------------------------------------------------------------------


def test_bm25_tokenizer():
    """Verify regex tokenizer extracts lowercase alphanumeric words."""
    tokens = tokenize_for_bm25("Apple Inc. recorded $90B in Q3-2023 revenue!")
    assert "apple" in tokens
    assert "revenue" in tokens
    assert "90b" in tokens


def test_bm25_indexing_and_search(retrieval_test_chunks: list[Chunk]):
    """Verify BM25 retrieves exact keyword matches with scores and provenance."""
    bm25 = BM25Index()
    indexed_count = bm25.index_chunks(retrieval_test_chunks)
    assert indexed_count == 4
    assert bm25.count() == 4

    results = bm25.search("iPhone revenue sales", top_k=2)
    assert len(results) > 0
    top = results[0]
    assert top.chunk_id == "doc1#stratA#c0000"
    assert top.document_id == "doc1"
    assert top.score > 0.0
    assert top.rank == 1
    assert top.retrieval_method == "bm25"
    assert top.page_numbers == [1]
    assert top.metadata["source"] == "apple_10q.pdf"


def test_bm25_empty_query_and_empty_index():
    """Verify BM25 handles empty queries and empty index gracefully."""
    empty_index = BM25Index()
    assert empty_index.search("test query") == []

    populated = BM25Index([
        Chunk(
            id="c1",
            document_id="d1",
            content="Hello world",
            token_count=2,
            chunk_index=0,
        )
    ])
    assert populated.search("") == []
    assert populated.search("   ") == []
    assert populated.search("completely_unrelated_xyz_token") == []


def test_bm25_duplicate_chunk_deduplication(retrieval_test_chunks: list[Chunk]):
    """Verify passing duplicate chunk IDs overwrites without inflating index."""
    chunks_with_duplicates = retrieval_test_chunks + [retrieval_test_chunks[0]]
    bm25 = BM25Index()
    count = bm25.index_chunks(chunks_with_duplicates)
    assert count == 4
    assert bm25.count() == 4


# ---------------------------------------------------------------------------
# 2. Dense Indexing & Retrieval Tests
# ---------------------------------------------------------------------------


def test_dense_indexing_and_search(
    tmp_path: Path, retrieval_test_chunks: list[Chunk]
):
    """Verify dense retrieval indexes chunks, queries cosine similarity, and restores metadata."""
    persist_dir = tmp_path / "test_chroma"
    dense = DenseIndex(persist_dir=persist_dir)

    indexed = dense.index_chunks(
        retrieval_test_chunks, collection_name="test_dense_col"
    )
    assert indexed == 4
    assert dense.count("test_dense_col") == 4

    results = dense.search(
        "Azure artificial intelligence and cloud computing",
        top_k=2,
        collection_name="test_dense_col",
    )
    assert len(results) == 2
    top = results[0]
    # Azure Cloud chunk should rank first
    assert top.chunk_id == "doc2#stratA#c0000"
    assert top.document_id == "doc2"
    assert top.retrieval_method == "dense"
    assert top.rank == 1
    assert 0.0 <= top.score <= 1.0
    assert top.page_numbers == [1]
    assert top.metadata["category"] == "cloud"


def test_dense_empty_and_top_k(tmp_path: Path):
    """Verify dense index handles empty collections and query bounds."""
    persist_dir = tmp_path / "test_empty_chroma"
    dense = DenseIndex(persist_dir=persist_dir)

    # Empty collection search
    assert dense.search("hello", collection_name="nonexistent") == []
    assert dense.search("", collection_name="nonexistent") == []


def test_dense_deduplication(
    tmp_path: Path, retrieval_test_chunks: list[Chunk]
):
    """Verify upserting duplicate chunk IDs deduplicates cleanly in ChromaDB."""
    persist_dir = tmp_path / "test_dedup_chroma"
    dense = DenseIndex(persist_dir=persist_dir)

    dense.index_chunks(retrieval_test_chunks, collection_name="dedup_col")
    # Index again with duplicates
    dense.index_chunks(retrieval_test_chunks, collection_name="dedup_col")
    assert dense.count("dedup_col") == 4


# ---------------------------------------------------------------------------
# 3. Hybrid / Reciprocal Rank Fusion Tests
# ---------------------------------------------------------------------------


def test_reciprocal_rank_fusion_logic():
    """Verify RRF correctly merges ranked candidate lists."""
    dense = ["docA", "docB", "docC"]
    sparse = ["docB", "docA", "docD"]

    fused = reciprocal_rank_fusion(dense, sparse, k=60)
    top_ids = [doc_id for doc_id, _ in fused]
    # Both docA and docB appear in top positions in both lists
    assert "docA" in top_ids[:2]
    assert "docB" in top_ids[:2]
    assert len(fused) == 4


def test_hybrid_retrieve_with_retrieved_chunks(
    retrieval_test_chunks: list[Chunk],
):
    """Verify hybrid_retrieve fuses candidates and preserves component ranks and provenance."""
    dense_cand = [
        RetrievedChunk(
            chunk_id="doc1#stratA#c0000",
            document_id="doc1",
            content="Apple revenue",
            score=0.92,
            rank=1,
            retrieval_method="dense",
            page_numbers=[1],
            metadata={"source": "apple.pdf"},
        ),
        RetrievedChunk(
            chunk_id="doc2#stratA#c0000",
            document_id="doc2",
            content="Microsoft cloud",
            score=0.75,
            rank=2,
            retrieval_method="dense",
            page_numbers=[1],
            metadata={"source": "msft.pdf"},
        ),
    ]

    sparse_cand = [
        RetrievedChunk(
            chunk_id="doc2#stratA#c0000",
            document_id="doc2",
            content="Microsoft cloud",
            score=4.5,
            rank=1,
            retrieval_method="bm25",
            page_numbers=[1],
            metadata={"source": "msft.pdf"},
        ),
        RetrievedChunk(
            chunk_id="doc1#stratA#c0001",
            document_id="doc1",
            content="Operating expenses",
            score=2.1,
            rank=2,
            retrieval_method="bm25",
            page_numbers=[2],
            metadata={"source": "apple.pdf"},
        ),
    ]

    fused = hybrid_retrieve(dense_cand, sparse_cand, k=60, top_k=2)
    assert len(fused) == 2

    # doc2 was rank 2 in dense and rank 1 in sparse -> boosted
    # doc1#c0000 was rank 1 in dense
    top_fused = fused[0]
    assert top_fused.retrieval_method == "hybrid"
    assert top_fused.rank == 1
    assert "rrf_dense_rank" in top_fused.metadata
    assert "rrf_sparse_rank" in top_fused.metadata


def test_hybrid_retrieve_empty_candidates():
    """Verify hybrid_retrieve handles empty input candidate lists safely."""
    assert hybrid_retrieve([], []) == []


# ---------------------------------------------------------------------------
# 4. Retriever Abstractions & Factory Configuration Tests
# ---------------------------------------------------------------------------


def test_retriever_factory_and_individual_modes(
    tmp_path: Path, retrieval_test_chunks: list[Chunk]
):
    """Verify dense, sparse, and hybrid retrievers are independently callable and configurable."""
    dense_index = DenseIndex(persist_dir=tmp_path / "factory_chroma")
    dense_index.index_chunks(retrieval_test_chunks, collection_name="rag_chunks")

    bm25_index = BM25Index()
    bm25_index.index_chunks(retrieval_test_chunks)

    # 1. Dense Only
    dense_retriever = create_retriever(
        strategy="dense_only",
        dense_index=dense_index,
        bm25_index=bm25_index,
    )
    assert isinstance(dense_retriever, DenseRetriever)
    dense_results = dense_retriever.retrieve("Apple iPhone revenue", top_k=2)
    assert len(dense_results) == 2
    assert all(r.retrieval_method == "dense" for r in dense_results)

    # 2. BM25 Only
    bm25_retriever = create_retriever(
        strategy="bm25_only",
        dense_index=dense_index,
        bm25_index=bm25_index,
    )
    assert isinstance(bm25_retriever, BM25Retriever)
    bm25_results = bm25_retriever.retrieve("cybersecurity risk", top_k=1)
    assert len(bm25_results) == 1
    assert bm25_results[0].chunk_id == "doc2#stratA#c0001"
    assert bm25_results[0].retrieval_method == "bm25"

    # 3. Hybrid
    hybrid_retriever = create_retriever(
        strategy="hybrid",
        dense_index=dense_index,
        bm25_index=bm25_index,
    )
    assert isinstance(hybrid_retriever, HybridRetriever)
    hybrid_results = hybrid_retriever.retrieve("Microsoft Azure AI", top_k=3)
    assert len(hybrid_results) >= 1
    assert hybrid_results[0].retrieval_method == "hybrid"


def test_retriever_factory_invalid_strategy():
    """Verify factory raises ValueError for unrecognized retrieval strategy."""
    with pytest.raises(ValueError, match="Unknown retrieval strategy"):
        create_retriever(strategy="unsupported_retriever")


# ---------------------------------------------------------------------------
# 5. Retrieval Evaluation Metrics Tests
# ---------------------------------------------------------------------------


def test_retrieval_metrics():
    """Verify Precision@K, Recall@K, and MRR calculations."""
    retrieved = ["chunk1", "chunk2", "chunk3", "chunk4", "chunk5"]
    ground_truth = {"chunk2", "chunk4", "chunk9"}

    # In top 3: chunk2 is relevant (1 out of 3)
    assert precision_at_k(retrieved, ground_truth, k=3) == 1 / 3
    # In top 5: chunk2 and chunk4 are relevant (2 out of 3 in ground truth)
    assert recall_at_k(retrieved, ground_truth, k=5) == 2 / 3
    # First relevant chunk is chunk2 at rank 2
    assert mean_reciprocal_rank(retrieved, ground_truth) == 0.5
