"""End-to-end pipeline orchestrating retrieval, reranking, and evidence-grounded generation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.config import RAGConfig, settings
from src.generation.generate import EvidenceGroundedGenerator, RAGResponse
from src.retrieval.retriever import BaseRetriever, create_retriever

if TYPE_CHECKING:
    pass


class RAGPipeline:
    """Orchestrates end-to-end query processing across retrieval and generation."""

    def __init__(
        self,
        retriever: BaseRetriever | None = None,
        generator: EvidenceGroundedGenerator | None = None,
        config: RAGConfig | None = None,
    ):
        self.config = config or settings
        self.retriever = retriever
        self.generator = generator or EvidenceGroundedGenerator(config=self.config)

    def query(
        self,
        question: str,
        retriever_mode: str | None = None,
        top_k: int | None = None,
    ) -> RAGResponse:
        """Execute RAG query with specified retriever mode and generate grounded response.

        Supported modes:
        - 'bm25_only'
        - 'dense_only'
        - 'hybrid'
        - 'hybrid_rerank'
        """
        active_mode = retriever_mode or self.config.retrieval_strategy
        retriever = self.retriever
        if retriever is None or (retriever_mode is not None):
            retriever = create_retriever(strategy=active_mode, config=self.config)

        # 1. Retrieve ranked candidates
        candidates = retriever.retrieve(question, top_k=top_k)

        # 2. Evidence-grounded generation with citation validation & refusal
        return self.generator.generate(question, retrieved_chunks=candidates)

