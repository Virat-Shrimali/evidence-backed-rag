"""Sparse keyword retrieval using Okapi BM25 over the chunk corpus."""

import re

from rank_bm25 import BM25Okapi

from src.ingest.chunk import Chunk
from src.retrieval.models import RetrievedChunk


def tokenize_for_bm25(text: str) -> list[str]:
    """Tokenize text into lowercase alphanumeric keywords for BM25 matching."""
    if not text:
        return []
    return [match.group(0).lower() for match in re.finditer(r"\b\w+\b", text)]


class BM25Index:
    """Sparse index wrapper using Rank-BM25 with full chunk provenance."""

    def __init__(self, chunks: list[Chunk] | None = None):
        self._chunks_map: dict[str, Chunk] = {}
        self._chunk_ids: list[str] = []
        self._corpus: list[list[str]] = []
        self._bm25: BM25Okapi | None = None

        if chunks:
            self.index_chunks(chunks)

    def index_chunks(self, chunks: list[Chunk]) -> int:
        """Build BM25 index over the provided chunks.

        Handles duplicate IDs by keeping the latest chunk.
        Returns the count of indexed chunks.
        """
        if not chunks:
            self.clear()
            return 0

        # Deduplicate chunks while preserving order
        unique_map: dict[str, Chunk] = {c.id: c for c in chunks}
        self._chunks_map = unique_map
        self._chunk_ids = list(unique_map.keys())

        # Tokenize corpus for BM25
        self._corpus = [tokenize_for_bm25(unique_map[cid].content) for cid in self._chunk_ids]

        if self._corpus:
            self._bm25 = BM25Okapi(self._corpus)
        else:
            self._bm25 = None

        return len(self._chunk_ids)

    def search(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        """Query BM25 index and return ranked chunks with complete provenance."""
        if not query.strip() or self._bm25 is None or not self._chunk_ids:
            return []

        tokenized_query = tokenize_for_bm25(query)
        if not tokenized_query:
            return []

        scores = self._bm25.get_scores(tokenized_query)
        scored_pairs = [
            (cid, float(score))
            for cid, score in zip(self._chunk_ids, scores, strict=False)
            if score > 0.0
        ]

        # Sort descending by BM25 score
        scored_pairs.sort(key=lambda x: x[1], reverse=True)

        k = max(1, top_k)
        top_candidates = scored_pairs[:k]

        results: list[RetrievedChunk] = []
        for rank, (cid, score) in enumerate(top_candidates):
            chunk = self._chunks_map[cid]
            results.append(
                RetrievedChunk(
                    chunk_id=chunk.id,
                    document_id=chunk.document_id,
                    content=chunk.content,
                    score=score,
                    rank=rank + 1,
                    retrieval_method="bm25",
                    page_numbers=list(chunk.page_numbers),
                    metadata=dict(chunk.metadata),
                )
            )

        return results

    def count(self) -> int:
        """Return the number of chunks currently indexed."""
        return len(self._chunk_ids)

    def clear(self) -> None:
        """Clear all indexed chunks."""
        self._chunks_map.clear()
        self._chunk_ids.clear()
        self._corpus.clear()
        self._bm25 = None
