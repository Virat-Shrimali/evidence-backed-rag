"""Unit tests for evaluation metrics, golden QA dataset, and comparison harness."""

from pathlib import Path

import pytest

from evaluation.build_golden_set import (
    GoldenQAPair,
    load_golden_qa_set,
    save_golden_qa_set,
    validate_golden_qa_set,
)
from evaluation.retrieval_metrics import (
    RetrievalEvaluationReport,
    evaluate_retriever,
    is_retrieval_refusal,
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from evaluation.run_comparison import generate_comparison_markdown_table
from src.retrieval.models import RetrievedChunk
from src.retrieval.retriever import BaseRetriever


def test_precision_at_k_basic():
    retrieved = ["c1", "c2", "c3", "c4", "c5"]
    ground_truth = {"c1", "c3"}

    assert precision_at_k(retrieved, ground_truth, k=5) == 2 / 5
    assert precision_at_k(retrieved, ground_truth, k=2) == 1 / 2
    assert precision_at_k(retrieved, ground_truth, k=1) == 1 / 1
    assert precision_at_k(retrieved, {"c99"}, k=5) == 0.0
    assert precision_at_k([], ground_truth, k=5) == 0.0
    assert precision_at_k(retrieved, set(), k=5) == 0.0
    assert precision_at_k(retrieved, ground_truth, k=0) == 0.0


def test_precision_at_k_with_retrieved_chunks():
    chunks = [
        RetrievedChunk(
            chunk_id="c1",
            document_id="d1",
            content="text 1",
            score=0.9,
            rank=1,
            retrieval_method="dense",
        ),
        RetrievedChunk(
            chunk_id="c2",
            document_id="d1",
            content="text 2",
            score=0.8,
            rank=2,
            retrieval_method="dense",
        ),
        RetrievedChunk(
            chunk_id="c3",
            document_id="d1",
            content="text 3",
            score=0.7,
            rank=3,
            retrieval_method="dense",
        ),
    ]
    ground_truth = ["c2"]
    assert precision_at_k(chunks, ground_truth, k=3) == pytest.approx(1 / 3)


def test_recall_at_k_basic():
    retrieved = ["c1", "c2", "c3", "c4", "c5"]
    ground_truth = {"c1", "c3"}

    assert recall_at_k(retrieved, ground_truth, k=5) == 1.0
    assert recall_at_k(retrieved, ground_truth, k=2) == 0.5
    assert recall_at_k(retrieved, {"c99", "c100"}, k=5) == 0.0
    assert recall_at_k([], ground_truth, k=5) == 0.0
    assert recall_at_k(retrieved, set(), k=5) == 0.0
    assert recall_at_k(retrieved, ground_truth, k=0) == 0.0


def test_reciprocal_rank():
    assert reciprocal_rank(["c1", "c2", "c3"], {"c1"}) == 1.0
    assert reciprocal_rank(["c2", "c1", "c3"], {"c1"}) == 0.5
    assert reciprocal_rank(["c3", "c2", "c1"], {"c1"}) == pytest.approx(1 / 3)
    assert reciprocal_rank(["c4", "c5", "c6"], {"c1"}) == 0.0
    assert reciprocal_rank([], {"c1"}) == 0.0
    assert reciprocal_rank(["c1"], set()) == 0.0
    assert mean_reciprocal_rank(["c2", "c1"], {"c1"}) == 0.5


def test_is_retrieval_refusal():
    assert is_retrieval_refusal([]) is True

    chunk_high = RetrievedChunk(
        chunk_id="c1",
        document_id="d1",
        content="ok",
        score=0.85,
        rank=1,
        retrieval_method="dense",
    )
    chunk_low = RetrievedChunk(
        chunk_id="c2",
        document_id="d1",
        content="low",
        score=0.35,
        rank=1,
        retrieval_method="dense",
    )

    assert is_retrieval_refusal([chunk_high], score_threshold=0.65) is False
    assert is_retrieval_refusal([chunk_low], score_threshold=0.65) is True

    chunk_rerank_low = RetrievedChunk(
        chunk_id="c3",
        document_id="d1",
        content="rr",
        score=0.9,
        rerank_score=-5.2,
        rank=1,
        retrieval_method="reranked",
    )
    chunk_rerank_high = RetrievedChunk(
        chunk_id="c4",
        document_id="d1",
        content="rr",
        score=0.9,
        rerank_score=3.5,
        rank=1,
        retrieval_method="reranked",
    )
    assert is_retrieval_refusal([chunk_rerank_low], score_threshold=0.0) is True
    assert is_retrieval_refusal([chunk_rerank_high], score_threshold=0.0) is False


def test_golden_qa_validation_rules():
    valid = [
        GoldenQAPair(
            id="qa_1",
            question="What is X?",
            expected_answer="X is Y.",
            relevant_chunk_ids=["c1"],
            is_answerable=True,
            difficulty="easy",
            category="overview",
        ),
        GoldenQAPair(
            id="qa_2",
            question="What is unanswerable?",
            expected_answer="Insufficient evidence",
            relevant_chunk_ids=[],
            is_answerable=False,
            difficulty="unanswerable",
            category="unanswerable",
        ),
    ]
    summary = validate_golden_qa_set(valid)
    assert summary["total_pairs"] == 2
    assert summary["answerable_count"] == 1
    assert summary["unanswerable_count"] == 1

    with pytest.raises(ValueError, match="empty"):
        validate_golden_qa_set([])

    with pytest.raises(ValueError, match="Duplicate"):
        validate_golden_qa_set([valid[0], valid[0]])

    with pytest.raises(ValueError, match="empty relevant_chunk_ids"):
        invalid_ans = GoldenQAPair(
            id="bad",
            question="Q",
            expected_answer="A",
            relevant_chunk_ids=[],
            is_answerable=True,
        )
        validate_golden_qa_set([invalid_ans])

    with pytest.raises(ValueError, match="should not have relevant_chunk_ids"):
        invalid_unans = GoldenQAPair(
            id="bad2",
            question="Q",
            expected_answer="A",
            relevant_chunk_ids=["c1"],
            is_answerable=False,
        )
        validate_golden_qa_set([invalid_unans])


def test_save_and_load_golden_qa_roundtrip(tmp_path: Path):
    target = tmp_path / "test_golden.jsonl"
    pairs = [
        GoldenQAPair(
            id="q1",
            question="Question 1?",
            expected_answer="Answer 1",
            relevant_chunk_ids=["c1"],
            is_answerable=True,
        )
    ]
    save_golden_qa_set(pairs, target)
    loaded = load_golden_qa_set(target)
    assert len(loaded) == 1
    assert loaded[0].id == "q1"
    assert loaded[0].question == "Question 1?"


def test_generate_comparison_markdown_table():
    reports = {
        "Strategy A": RetrievalEvaluationReport(
            strategy_name="Strategy A",
            recall_at_k=0.85,
            precision_at_k=0.20,
            mrr=0.60,
            refusal_accuracy=1.0,
            latency_p50_ms=12.5,
            latency_p95_ms=25.0,
        )
    }
    table = generate_comparison_markdown_table(reports, top_k=5)
    assert "| Retriever Strategy | Recall@5 | Precision@5 | MRR | Refusal Accuracy | p50 Latency (ms) | p95 Latency (ms) |" in table
    assert "**Strategy A**" in table
    assert "85.00%" in table
    assert "20.00%" in table
    assert "100.00%" in table


class MockRetriever(BaseRetriever):
    def __init__(self, mapping: dict[str, list[RetrievedChunk]]):
        self.mapping = mapping

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        return self.mapping.get(query, [])


def test_evaluate_retriever_mock():
    c1 = RetrievedChunk(
        chunk_id="c1",
        document_id="d1",
        content="text",
        score=0.85,
        rank=1,
        retrieval_method="mock",
    )
    c2 = RetrievedChunk(
        chunk_id="c2",
        document_id="d1",
        content="text",
        score=0.40,
        rank=1,
        retrieval_method="mock",
    )

    mock = MockRetriever({
        "What is A?": [c1],
        "What is out of domain?": [c2],
    })

    dataset = [
        GoldenQAPair(
            id="1",
            question="What is A?",
            expected_answer="Ans A",
            relevant_chunk_ids=["c1"],
            is_answerable=True,
        ),
        GoldenQAPair(
            id="2",
            question="What is out of domain?",
            expected_answer="Refuse",
            relevant_chunk_ids=[],
            is_answerable=False,
        ),
    ]

    report = evaluate_retriever(
        mock, dataset, top_k=5, score_threshold=0.60, strategy_name="MockStrategy"
    )
    assert report.strategy_name == "MockStrategy"
    assert report.total_queries == 2
    assert report.answerable_count == 1
    assert report.unanswerable_count == 1
    assert report.recall_at_k == 1.0
    assert report.precision_at_k == 0.20  # 1 relevant chunk out of 5 top_k slots
    assert report.mrr == 1.0
    assert report.refusal_accuracy == 1.0  # c2 score 0.40 < 0.60 threshold -> refused
