"""Hybrid retrieval using Reciprocal Rank Fusion (RRF)."""



def reciprocal_rank_fusion(
    dense_results: list[str],
    sparse_results: list[str],
    k: int = 60,
) -> list[tuple[str, float]]:
    """Merge ranked results from dense and sparse retrievers using RRF.

    score(d) = sum(1 / (k + rank + 1))
    """
    scores: dict[str, float] = {}

    for rank, doc_id in enumerate(dense_results):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)

    for rank, doc_id in enumerate(sparse_results):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)

    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


def hybrid_retrieve(
    dense_results: list[str],
    sparse_results: list[str],
    k: int = 60,
    top_k: int = 5,
) -> list[str]:
    """Retrieve top_k document IDs using Reciprocal Rank Fusion."""
    fused = reciprocal_rank_fusion(dense_results, sparse_results, k=k)
    return [doc_id for doc_id, _ in fused[:top_k]]
