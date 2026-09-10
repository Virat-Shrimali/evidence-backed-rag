"""Comprehensive unit tests for the CrossEncoderReranker and RerankedRetriever pipeline."""

from unittest.mock import MagicMock

import pytest

from src.retrieval.models import RetrievedChunk
from src.retrieval.rerank import CrossEncoderReranker
from src.retrieval.retriever import (
    BaseRetriever,
    RerankedRetriever,
    create_retriever,
)


@pytest.fixture
def sample_candidates() -> list[RetrievedChunk]:
    """Sample candidates produced by Stage 1 retrieval (dense, BM25, or hybrid)."""
    return [
        RetrievedChunk(
            chunk_id="doc1#c0001",
            document_id="doc1",
            content="Apple reported quarterly revenue of 90 billion dollars.",
            score=0.75,
            rank=1,
            retrieval_method="hybrid",
            page_numbers=[1],
            metadata={"source": "apple.pdf", "section": "revenue"},
        ),
        RetrievedChunk(
            chunk_id="doc1#c0002",
            document_id="doc1",
            content="Operating expenses increased due to R&D investments.",
            score=0.68,
            rank=2,
            retrieval_method="hybrid",
            page_numbers=[1, 2],
            metadata={"source": "apple.pdf", "section": "expenses"},
        ),
        RetrievedChunk(
            chunk_id="doc2#c0001",
            document_id="doc2",
            content="Microsoft Cloud revenue expanded to 35 billion dollars.",
            score=0.62,
            rank=3,
            retrieval_method="hybrid",
            page_numbers=[3],
            metadata={"source": "msft.pdf", "section": "cloud"},
        ),
        RetrievedChunk(
            chunk_id="doc3#c0001",
            document_id="doc3",
            content="General inflation affects consumer electronic product margins.",
            score=0.55,
            rank=4,
            retrieval_method="hybrid",
            page_numbers=[1],
            metadata={"source": "macro.pdf", "section": "economy"},
        ),
    ]


class MockCrossEncoderModel:
    """Mock cross-encoder that returns predetermined scores for fast, offline testing."""

    def __init__(self, score_map: dict[str, float] | None = None):
        self.score_map = score_map or {}

    def predict(self, pairs: list[list[str]]) -> list[float]:
        results = []
        for _query, content in pairs:
            # Check if snippet contains matching keyword in score_map
            score = 0.1
            for key, val in self.score_map.items():
                if key.lower() in content.lower():
                    score = val
                    break
            results.append(score)
        return results


# ---------------------------------------------------------------------------
# 1. Reranker Ranking and Provenance Preservation Tests
# ---------------------------------------------------------------------------


def test_reranker_reorders_and_preserves_provenance(
    sample_candidates: list[RetrievedChunk],
):
    """Verify reranker reorders candidates based on neural scores and preserves original fields."""
    # Chunk doc2#c0001 (Microsoft Cloud) was rank 3 initially; mock model gives it top score 4.5
    mock_model = MockCrossEncoderModel(
        score_map={"Microsoft Cloud": 4.5, "Apple reported": 2.0, "Operating": 0.5}
    )
    reranker = CrossEncoderReranker(
        model_name="mock-cross-encoder", model=mock_model
    )

    results = reranker.rerank(
        query="Microsoft cloud infrastructure performance",
        candidates=sample_candidates,
        top_k=3,
    )

    assert len(results) == 3

    # Rank 1 should now be doc2#c0001
    top_chunk = results[0]
    assert top_chunk.chunk_id == "doc2#c0001"
    assert top_chunk.document_id == "doc2"
    assert top_chunk.rerank_rank == 1
    assert top_chunk.rerank_score == 4.5

    # CRITICAL: Verify original score and rank are preserved and NOT overwritten
    assert top_chunk.score == 0.62
    assert top_chunk.rank == 3
    assert top_chunk.original_score == 0.62
    assert top_chunk.original_rank == 3

    # Verify provenance and metadata preservation
    assert top_chunk.page_numbers == [3]
    assert top_chunk.metadata["source"] == "msft.pdf"
    assert top_chunk.metadata["section"] == "cloud"
    assert top_chunk.metadata["reranker_model"] == "mock-cross-encoder"
    assert top_chunk.metadata["stage1_retrieval_method"] == "hybrid"


def test_reranker_top_k_truncation(sample_candidates: list[RetrievedChunk]):
    """Verify reranker accurately limits output to requested top_k."""
    mock_model = MockCrossEncoderModel()
    reranker = CrossEncoderReranker(model=mock_model)

    top_2 = reranker.rerank("query", sample_candidates, top_k=2)
    assert len(top_2) == 2
    assert top_2[0].rerank_rank == 1
    assert top_2[1].rerank_rank == 2

    # Fewer candidates than top_k
    top_10 = reranker.rerank("query", sample_candidates, top_k=10)
    assert len(top_10) == 4


# ---------------------------------------------------------------------------
# 2. Edge Cases and Deduplication Tests
# ---------------------------------------------------------------------------


def test_reranker_empty_inputs(sample_candidates: list[RetrievedChunk]):
    """Verify reranker handles empty queries, empty candidates, or non-positive top_k."""
    reranker = CrossEncoderReranker(model=MockCrossEncoderModel())

    assert reranker.rerank("", sample_candidates, top_k=3) == []
    assert reranker.rerank("   ", sample_candidates, top_k=3) == []
    assert reranker.rerank("valid query", [], top_k=3) == []
    assert reranker.rerank("valid query", sample_candidates, top_k=0) == []
    assert reranker.rerank("valid query", sample_candidates, top_k=-5) == []


def test_reranker_duplicate_chunk_handling(
    sample_candidates: list[RetrievedChunk],
):
    """Verify candidate list containing duplicate chunk IDs is deduplicated cleanly."""
    duplicate_cand = RetrievedChunk(
        chunk_id="doc1#c0001",
        document_id="doc1",
        content="Apple reported quarterly revenue of 90 billion dollars.",
        score=0.90,  # Higher score
        rank=1,
        retrieval_method="dense",
        page_numbers=[1],
        metadata={"source": "apple.pdf"},
    )
    candidates_with_dups = [sample_candidates[0], duplicate_cand, sample_candidates[1]]

    reranker = CrossEncoderReranker(model=MockCrossEncoderModel())
    results = reranker.rerank("query", candidates_with_dups, top_k=5)

    # Should have 2 unique chunk IDs (doc1#c0001 and doc1#c0002)
    ids = [r.chunk_id for r in results]
    assert len(ids) == len(set(ids))
    assert len(results) == 2


def test_reranker_deterministic_tie_breaking():
    """Verify deterministic tie-breaking by original rank and chunk_id when scores are equal."""
    cands = [
        RetrievedChunk(
            chunk_id="docB#c0001",
            document_id="docB",
            content="Text B",
            score=0.5,
            rank=2,
            retrieval_method="bm25",
        ),
        RetrievedChunk(
            chunk_id="docA#c0001",
            document_id="docA",
            content="Text A",
            score=0.8,
            rank=1,
            retrieval_method="bm25",
        ),
    ]

    # Both get score 1.0
    mock_model = MagicMock()
    mock_model.predict.return_value = [1.0, 1.0]

    reranker = CrossEncoderReranker(model=mock_model)
    results = reranker.rerank("query", cands, top_k=2)

    # docA#c0001 has rank 1, docB has rank 2 -> docA ranks first on tie
    assert results[0].chunk_id == "docA#c0001"
    assert results[1].chunk_id == "docB#c0001"


# ---------------------------------------------------------------------------
# 3. Interface Compatibility with BM25, Dense, and Hybrid
# ---------------------------------------------------------------------------


def test_reranker_works_with_bm25_dense_and_hybrid():
    """Verify reranker can accept candidates from BM25, dense, or hybrid retrievers."""
    bm25_cand = [
        RetrievedChunk(
            chunk_id="bm25_1",
            document_id="d1",
            content="Keyword match",
            score=5.2,
            rank=1,
            retrieval_method="bm25",
        )
    ]
    dense_cand = [
        RetrievedChunk(
            chunk_id="dense_1",
            document_id="d2",
            content="Semantic match",
            score=0.88,
            rank=1,
            retrieval_method="dense",
        )
    ]

    reranker = CrossEncoderReranker(model=MockCrossEncoderModel())

    bm25_reranked = reranker.rerank("query", bm25_cand)
    assert len(bm25_reranked) == 1
    assert bm25_reranked[0].retrieval_method == "bm25"
    assert bm25_reranked[0].original_score == 5.2

    dense_reranked = reranker.rerank("query", dense_cand)
    assert len(dense_reranked) == 1
    assert dense_reranked[0].retrieval_method == "dense"
    assert dense_reranked[0].original_score == 0.88


# ---------------------------------------------------------------------------
# 4. Production Pipeline: RerankedRetriever (Top-20 -> Top-5)
# ---------------------------------------------------------------------------


def test_reranked_retriever_production_pipeline():
    """Verify RerankedRetriever implements the BM25+Dense -> RRF -> Top-20 -> Cross-Encoder -> Top-5 pipeline."""
    # Create mock base retriever returning 10 candidates
    mock_base = MagicMock(spec=BaseRetriever)
    mock_base.retrieve.return_value = [
        RetrievedChunk(
            chunk_id=f"chunk_{i}",
            document_id=f"doc_{i}",
            content=f"Candidate text {i}",
            score=1.0 / (i + 1),
            rank=i + 1,
            retrieval_method="hybrid",
        )
        for i in range(10)
    ]

    mock_reranker = CrossEncoderReranker(
        model=MockCrossEncoderModel(score_map={"Candidate text 7": 9.9})
    )

    pipeline_retriever = RerankedRetriever(
        base_retriever=mock_base,
        reranker=mock_reranker,
        candidate_top_k=10,
        final_top_k=3,
    )

    results = pipeline_retriever.retrieve("my search query", top_k=3)

    # Base retriever requested candidate_top_k
    mock_base.retrieve.assert_called_once_with("my search query", top_k=10)
    assert len(results) == 3
    # Candidate 7 boosted to top
    assert results[0].chunk_id == "chunk_7"
    assert results[0].rerank_rank == 1
    assert results[0].rerank_score == 9.9


def test_create_retriever_factory_hybrid_rerank():
    """Verify create_retriever factory creates RerankedRetriever when strategy is hybrid_rerank."""
    mock_reranker = CrossEncoderReranker(model=MockCrossEncoderModel())
    retriever = create_retriever(
        strategy="hybrid_rerank",
        reranker=mock_reranker,
    )
    assert isinstance(retriever, RerankedRetriever)
    assert retriever.candidate_top_k == 20
    assert retriever.final_top_k == 5


# ---------------------------------------------------------------------------
# 5. Real Model Integration Smoke Test
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_real_cross_encoder_smoke_test():
    """Integration test using the real sentence-transformers CrossEncoder model."""
    reranker = CrossEncoderReranker(
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    candidates = [
        RetrievedChunk(
            chunk_id="c1",
            document_id="d1",
            content="The capital of France is Paris.",
            score=0.5,
            rank=1,
            retrieval_method="dense",
        ),
        RetrievedChunk(
            chunk_id="c2",
            document_id="d2",
            content="Apples grow on deciduous trees.",
            score=0.4,
            rank=2,
            retrieval_method="dense",
        ),
    ]

    results = reranker.rerank(
        query="What is the capital city of France?",
        candidates=candidates,
        top_k=2,
    )
    assert len(results) == 2
    # Paris chunk must score higher than apples
    assert results[0].chunk_id == "c1"
    assert results[0].rerank_rank == 1
    assert results[0].rerank_score > results[1].rerank_score
