"""Unit tests for retrieval fusion and evaluation metrics."""

from evaluation.retrieval_metrics import mean_reciprocal_rank, precision_at_k, recall_at_k
from src.retrieval.hybrid import hybrid_retrieve, reciprocal_rank_fusion


def test_reciprocal_rank_fusion():
    """Verify RRF correctly boosts documents ranked high in both dense and sparse."""
    dense_results = ["docA", "docB", "docC"]
    sparse_results = ["docB", "docA", "docD"]

    fused = reciprocal_rank_fusion(dense_results, sparse_results, k=60)
    # Both docA and docB appear in top ranks across both lists
    fused_ids = [doc_id for doc_id, score in fused]
    assert "docA" in fused_ids[:2]
    assert "docB" in fused_ids[:2]


def test_hybrid_retrieve_top_k():
    """Verify hybrid_retrieve limits results to top_k."""
    dense_results = ["doc1", "doc2", "doc3", "doc4", "doc5"]
    sparse_results = ["doc3", "doc2", "doc6", "doc7"]

    top_3 = hybrid_retrieve(dense_results, sparse_results, top_k=3)
    assert len(top_3) == 3


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
