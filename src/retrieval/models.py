"""Shared data models for retrieval results and provenance."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RetrievedChunk:
    """A retrieved chunk candidate with ranking score and complete provenance."""
    chunk_id: str
    document_id: str
    content: str
    score: float
    rank: int
    retrieval_method: str  # "dense", "bm25", or "hybrid"
    page_numbers: list[int] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    rerank_score: float | None = None
    rerank_rank: int | None = None
    original_score: float | None = None
    original_rank: int | None = None
