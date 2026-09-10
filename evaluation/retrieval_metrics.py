"""Retrieval evaluation metrics: Precision@K, Recall@K, and MRR."""



def precision_at_k(retrieved: list[str], ground_truth: set[str], k: int = 5) -> float:
    """Compute Precision@K.

    Precision@K = (# of relevant retrieved chunks in top K) / K
    """
    if k <= 0:
        return 0.0
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    relevant_retrieved = sum(1 for item in top_k if item in ground_truth)
    return relevant_retrieved / k


def recall_at_k(retrieved: list[str], ground_truth: set[str], k: int = 5) -> float:
    """Compute Recall@K.

    Recall@K = (# of relevant retrieved chunks in top K) / (total relevant chunks)
    """
    if not ground_truth:
        return 0.0
    top_k = retrieved[:k]
    relevant_retrieved = sum(1 for item in top_k if item in ground_truth)
    return relevant_retrieved / len(ground_truth)


def mean_reciprocal_rank(retrieved: list[str], ground_truth: set[str]) -> float:
    """Compute Reciprocal Rank (RR) for a single query.

    RR = 1 / rank_of_first_relevant_chunk (or 0.0 if none found)
    """
    for rank, item in enumerate(retrieved, start=1):
        if item in ground_truth:
            return 1.0 / rank
    return 0.0
