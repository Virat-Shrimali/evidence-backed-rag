"""Retrieval evaluation metrics: Precision@K, Recall@K, MRR, Refusal Accuracy, and Latency."""

from __future__ import annotations

import statistics
import time
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from evaluation.build_golden_set import GoldenQAPair
    from src.retrieval.models import RetrievedChunk
    from src.retrieval.retriever import BaseRetriever


def extract_chunk_ids(retrieved: list[str] | list[RetrievedChunk] | list[Any]) -> list[str]:
    """Extract string chunk IDs from a list of strings or RetrievedChunk objects."""
    if not retrieved:
        return []
    ids: list[str] = []
    for item in retrieved:
        if isinstance(item, str):
            ids.append(item)
        elif hasattr(item, "chunk_id"):
            ids.append(item.chunk_id)
        elif hasattr(item, "id"):
            ids.append(item.id)
        else:
            ids.append(str(item))
    return ids


def precision_at_k(
    retrieved: list[str] | list[RetrievedChunk] | list[Any],
    ground_truth: set[str] | list[str],
    k: int = 5,
) -> float:
    """Compute Precision@K.

    Precision@K = (# of relevant retrieved chunks in top K) / K
    """
    if k <= 0:
        return 0.0
    ids = extract_chunk_ids(retrieved)[:k]
    if not ids:
        return 0.0
    gt_set = set(ground_truth)
    if not gt_set:
        return 0.0
    relevant_retrieved = sum(1 for item in ids if item in gt_set)
    return relevant_retrieved / k


def recall_at_k(
    retrieved: list[str] | list[RetrievedChunk] | list[Any],
    ground_truth: set[str] | list[str],
    k: int = 5,
) -> float:
    """Compute Recall@K.

    Recall@K = (# of relevant retrieved chunks in top K) / (total relevant chunks)
    """
    gt_set = set(ground_truth)
    if not gt_set:
        return 0.0
    if k <= 0:
        return 0.0
    ids = extract_chunk_ids(retrieved)[:k]
    relevant_retrieved = sum(1 for item in ids if item in gt_set)
    return relevant_retrieved / len(gt_set)


def reciprocal_rank(
    retrieved: list[str] | list[RetrievedChunk] | list[Any],
    ground_truth: set[str] | list[str],
) -> float:
    """Compute Reciprocal Rank (RR) for a single query.

    RR = 1 / rank_of_first_relevant_chunk (or 0.0 if none found)
    """
    gt_set = set(ground_truth)
    if not gt_set:
        return 0.0
    ids = extract_chunk_ids(retrieved)
    for rank, item in enumerate(ids, start=1):
        if item in gt_set:
            return 1.0 / rank
    return 0.0


def mean_reciprocal_rank(
    retrieved: list[str] | list[RetrievedChunk] | list[Any],
    ground_truth: set[str] | list[str],
) -> float:
    """Compute Reciprocal Rank for a query (backwards-compatible alias)."""
    return reciprocal_rank(retrieved, ground_truth)


def is_retrieval_refusal(
    retrieved: list[RetrievedChunk] | list[Any],
    score_threshold: float | None = None,
) -> bool:
    """Determine whether the retrieved candidates trigger a retrieval refusal.

    A refusal is triggered if:
    1. No chunks were retrieved at all (empty candidate list).
    2. A score threshold is specified and the top candidate chunk's score is below threshold.
    """
    if not retrieved:
        return True

    if score_threshold is not None:
        top_chunk = retrieved[0]
        # Check rerank_score if available, otherwise base score
        if hasattr(top_chunk, "rerank_score") and top_chunk.rerank_score is not None:
            return top_chunk.rerank_score < score_threshold
        elif hasattr(top_chunk, "score") and top_chunk.score is not None:
            return top_chunk.score < score_threshold

    return False


class RetrievalEvaluationReport(BaseModel):
    """Aggregated evaluation metrics for a single retriever strategy."""

    strategy_name: str
    recall_at_k: float = 0.0
    precision_at_k: float = 0.0
    mrr: float = 0.0
    refusal_accuracy: float = 0.0
    latency_mean_ms: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    total_queries: int = 0
    answerable_count: int = 0
    unanswerable_count: int = 0
    query_metrics: list[dict[str, Any]] = Field(default_factory=list)


def evaluate_retriever(
    retriever: BaseRetriever,
    dataset: list[GoldenQAPair],
    top_k: int = 5,
    score_threshold: float | None = None,
    strategy_name: str = "retriever",
) -> RetrievalEvaluationReport:
    """Run an evaluation over a golden QA dataset using the given retriever."""
    if not dataset:
        return RetrievalEvaluationReport(strategy_name=strategy_name)

    recalls: list[float] = []
    precisions: list[float] = []
    mrrs: list[float] = []
    latencies_ms: list[float] = []
    refusal_correct_count = 0
    unanswerable_total = 0
    answerable_total = 0
    query_metrics: list[dict[str, Any]] = []

    for pair in dataset:
        start_t = time.perf_counter()
        retrieved_chunks = retriever.retrieve(pair.question, top_k=top_k)
        elapsed_ms = (time.perf_counter() - start_t) * 1000.0
        latencies_ms.append(elapsed_ms)

        retrieved_ids = extract_chunk_ids(retrieved_chunks)

        if pair.is_answerable:
            answerable_total += 1
            rec = recall_at_k(retrieved_ids, pair.relevant_chunk_ids, k=top_k)
            prec = precision_at_k(retrieved_ids, pair.relevant_chunk_ids, k=top_k)
            rr = reciprocal_rank(retrieved_ids, pair.relevant_chunk_ids)

            recalls.append(rec)
            precisions.append(prec)
            mrrs.append(rr)

            query_metrics.append({
                "id": pair.id,
                "question": pair.question,
                "is_answerable": True,
                "recall": rec,
                "precision": prec,
                "mrr": rr,
                "latency_ms": round(elapsed_ms, 2),
                "retrieved_ids": retrieved_ids[:top_k],
            })
        else:
            unanswerable_total += 1
            # Check if retriever correctly indicated refusal / lack of evidence
            refused = is_retrieval_refusal(retrieved_chunks, score_threshold=score_threshold)
            if refused:
                refusal_correct_count += 1

            query_metrics.append({
                "id": pair.id,
                "question": pair.question,
                "is_answerable": False,
                "refused": refused,
                "latency_ms": round(elapsed_ms, 2),
                "retrieved_ids": retrieved_ids[:top_k],
            })

    mean_recall = statistics.mean(recalls) if recalls else 0.0
    mean_precision = statistics.mean(precisions) if precisions else 0.0
    mean_mrr = statistics.mean(mrrs) if mrrs else 0.0
    refusal_acc = (
        (refusal_correct_count / unanswerable_total) if unanswerable_total > 0 else 1.0
    )

    lat_sorted = sorted(latencies_ms) if latencies_ms else [0.0]
    p50_idx = int(0.50 * len(lat_sorted))
    p95_idx = min(int(0.95 * len(lat_sorted)), len(lat_sorted) - 1)

    return RetrievalEvaluationReport(
        strategy_name=strategy_name,
        recall_at_k=round(mean_recall, 4),
        precision_at_k=round(mean_precision, 4),
        mrr=round(mean_mrr, 4),
        refusal_accuracy=round(refusal_acc, 4),
        latency_mean_ms=round(statistics.mean(latencies_ms) if latencies_ms else 0.0, 2),
        latency_p50_ms=round(lat_sorted[p50_idx], 2),
        latency_p95_ms=round(lat_sorted[p95_idx], 2),
        total_queries=len(dataset),
        answerable_count=answerable_total,
        unanswerable_count=unanswerable_total,
        query_metrics=query_metrics,
    )

