"""Hybrid retrieval combining dense and sparse results via Reciprocal Rank Fusion (RRF)."""

from typing import Any

from src.retrieval.models import RetrievedChunk


def reciprocal_rank_fusion_scores(
    rankings: list[list[str]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """Compute Reciprocal Rank Fusion scores across multiple ranked lists of document/chunk IDs.

    Formula: score(d) = sum(1 / (k + rank + 1))
    """
    scores: dict[str, float] = {}
    for ranked_list in rankings:
        for rank, doc_id in enumerate(ranked_list):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)

    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


def reciprocal_rank_fusion(
    dense_results: list[str],
    sparse_results: list[str],
    k: int = 60,
) -> list[tuple[str, float]]:
    """Backward-compatible helper merging dense and sparse ID lists."""
    return reciprocal_rank_fusion_scores([dense_results, sparse_results], k=k)


def hybrid_retrieve(
    dense_results: list[RetrievedChunk],
    sparse_results: list[RetrievedChunk],
    k: int = 60,
    top_k: int = 5,
) -> list[RetrievedChunk]:
    """Fuse dense and sparse RetrievedChunk candidates using Reciprocal Rank Fusion.

    - Resolves duplicates across retrievers.
    - Preserves complete chunk content, page numbers, and provenance.
    - Records dense and sparse ranks in metadata for auditing and transparency.
    """
    if not dense_results and not sparse_results:
        return []

    # Backward compatibility when string doc_ids are passed directly
    if (dense_results and isinstance(dense_results[0], str)) or (
        sparse_results and isinstance(sparse_results[0], str)
    ):
        fused = reciprocal_rank_fusion(
            dense_results,  # type: ignore[arg-type]
            sparse_results,  # type: ignore[arg-type]
            k=k,
        )
        return [doc_id for doc_id, _ in fused[:top_k]]  # type: ignore[return-value]

    # Map chunk_id to candidate objects and track component ranks
    chunk_store: dict[str, RetrievedChunk] = {}
    dense_ranks: dict[str, int] = {}
    sparse_ranks: dict[str, int] = {}

    dense_ids: list[str] = []
    for chunk in dense_results:
        dense_ids.append(chunk.chunk_id)
        dense_ranks[chunk.chunk_id] = chunk.rank
        chunk_store[chunk.chunk_id] = chunk

    sparse_ids: list[str] = []
    for chunk in sparse_results:
        sparse_ids.append(chunk.chunk_id)
        sparse_ranks[chunk.chunk_id] = chunk.rank
        if chunk.chunk_id not in chunk_store:
            chunk_store[chunk.chunk_id] = chunk

    # Compute RRF across dense and sparse ranking lists
    fused_scores = reciprocal_rank_fusion_scores([dense_ids, sparse_ids], k=k)

    fused_results: list[RetrievedChunk] = []
    for rank_idx, (chunk_id, rrf_score) in enumerate(fused_scores[: max(1, top_k)]):
        base_chunk = chunk_store[chunk_id]
        fused_meta: dict[str, Any] = dict(base_chunk.metadata)
        fused_meta["rrf_dense_rank"] = dense_ranks.get(chunk_id)
        fused_meta["rrf_sparse_rank"] = sparse_ranks.get(chunk_id)

        fused_results.append(
            RetrievedChunk(
                chunk_id=base_chunk.chunk_id,
                document_id=base_chunk.document_id,
                content=base_chunk.content,
                score=round(rrf_score, 6),
                rank=rank_idx + 1,
                retrieval_method="hybrid",
                page_numbers=list(base_chunk.page_numbers),
                metadata=fused_meta,
            )
        )

    return fused_results
