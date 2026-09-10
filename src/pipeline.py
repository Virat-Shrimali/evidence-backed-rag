"""End-to-end pipeline orchestrating retrieval, reranking, and evidence-grounded generation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.config import RAGConfig, settings
from src.generation.generate import EvidenceGroundedGenerator, RAGResponse
from src.retrieval.retriever import BaseRetriever, create_retriever

if TYPE_CHECKING:
    from src.retrieval.models import RetrievedChunk


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
        self._retrievers_by_mode: dict[str, BaseRetriever] = {}

    def get_retriever(self, retriever_mode: str | None = None) -> BaseRetriever:
        """Get or lazily create and cache retriever for the specified strategy."""
        if self.retriever is not None:
            return self.retriever
        active_mode = retriever_mode or self.config.retrieval_strategy
        if active_mode not in self._retrievers_by_mode:
            self._retrievers_by_mode[active_mode] = create_retriever(
                strategy=active_mode, config=self.config
            )
        return self._retrievers_by_mode[active_mode]

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
        response, _ = self.query_with_candidates(
            question=question,
            retriever_mode=retriever_mode,
            top_k=top_k,
        )
        return response

    def query_with_candidates(
        self,
        question: str,
        retriever_mode: str | None = None,
        top_k: int | None = None,
    ) -> tuple[RAGResponse, list[RetrievedChunk]]:
        """Execute RAG query and return both structured response and ranked candidate chunks."""
        retriever = self.get_retriever(retriever_mode)
        active_mode = retriever_mode or self.config.retrieval_strategy

        # For bm25_only retrieval in the pipeline, ensure the candidate pool
        # provided to generation receives the configured sparse_top_k so that
        # valid evidence beyond the top few keyword matches is visible.
        if active_mode == "bm25_only":
            retrieve_k = max(top_k or 0, self.config.sparse_top_k)
        else:
            retrieve_k = top_k

        # 1. Retrieve ranked candidates
        candidates = retriever.retrieve(question, top_k=retrieve_k)

        # 2. Evidence-grounded generation with citation validation & refusal
        response = self.generator.generate(question, retrieved_chunks=candidates)
        return response, candidates

