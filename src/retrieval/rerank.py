"""Independent Cross-Encoder reranking module (Tier S)."""

from typing import Any

from src.config import settings
from src.retrieval.models import RetrievedChunk


class CrossEncoderReranker:
    """Second-stage neural reranker using cross-encoder models."""

    def __init__(
        self,
        model_name: str | None = None,
        model: Any | None = None,
    ):
        self.model_name = model_name or settings.reranker_model_name
        self._model = model

    def _load_model(self) -> Any:
        """Lazily load SentenceTransformers CrossEncoder model."""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(self.model_name)
            except ImportError:
                self._model = None
        return self._model

    def rerank(
        self,
        query: str,
        candidates: list[RetrievedChunk],
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        """Score and rerank candidate chunks against the query.

        - Preserves original retrieval score and rank without overwriting.
        - Sets rerank_score and rerank_rank separately.
        - Preserves all metadata, page numbers, and chunk IDs.
        - Handles empty queries, empty candidates, duplicates, and tied scores.
        """
        if not query.strip() or not candidates or top_k <= 0:
            return []

        # Deduplicate candidates by chunk_id, preserving the candidate with the highest initial score
        unique_candidates_map: dict[str, RetrievedChunk] = {}
        for cand in candidates:
            if cand.chunk_id not in unique_candidates_map:
                unique_candidates_map[cand.chunk_id] = cand
            else:
                existing = unique_candidates_map[cand.chunk_id]
                if cand.score > existing.score:
                    unique_candidates_map[cand.chunk_id] = cand

        deduped_candidates = list(unique_candidates_map.values())
        if not deduped_candidates:
            return []

        model = self._load_model()
        if model is not None:
            pairs = [[query, chunk.content] for chunk in deduped_candidates]
            raw_scores = model.predict(pairs)
            scores = [float(s) for s in raw_scores]
        else:
            # Fallback deterministic scoring if model cannot be loaded
            scores = [1.0 / (idx + 1) for idx in range(len(deduped_candidates))]

        # Associate scores and sort deterministically:
        # 1. Primary: descending rerank_score
        # 2. Secondary tie-breaker: original retrieval rank (ascending)
        # 3. Tertiary tie-breaker: chunk_id (lexicographical)
        scored_pairs = list(zip(deduped_candidates, scores, strict=True))
        scored_pairs.sort(
            key=lambda item: (-item[1], item[0].rank, item[0].chunk_id)
        )

        k = min(top_k, len(scored_pairs))
        reranked_results: list[RetrievedChunk] = []

        for new_rank, (orig_cand, rerank_score) in enumerate(
            scored_pairs[:k], start=1
        ):
            meta = dict(orig_cand.metadata)
            meta["reranker_model"] = self.model_name
            meta["stage1_retrieval_method"] = orig_cand.retrieval_method
            meta["stage1_score"] = orig_cand.score
            meta["stage1_rank"] = orig_cand.rank

            reranked_results.append(
                RetrievedChunk(
                    chunk_id=orig_cand.chunk_id,
                    document_id=orig_cand.document_id,
                    content=orig_cand.content,
                    # Original retrieval score and rank are preserved, NOT overwritten
                    score=orig_cand.score,
                    rank=orig_cand.rank,
                    retrieval_method=orig_cand.retrieval_method,
                    page_numbers=list(orig_cand.page_numbers),
                    metadata=meta,
                    # Rerank score and rank are added separately
                    rerank_score=round(rerank_score, 6),
                    rerank_rank=new_rank,
                    original_score=orig_cand.score,
                    original_rank=orig_cand.rank,
                )
            )

        return reranked_results
