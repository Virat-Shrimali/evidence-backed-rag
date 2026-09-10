"""End-to-end pipeline orchestrating retrieval, reranking, and generation."""

from src.config import settings
from src.generation.generate import RAGResponse, make_refusal_response


class RAGPipeline:
    """Orchestrates query processing across retrieval and generation."""

    def __init__(self):
        self.settings = settings

    def query(
        self,
        question: str,
        retriever_mode: str = "hybrid_rerank",
    ) -> RAGResponse:
        """Execute RAG query with specified retriever mode.

        Supported modes:
        - 'bm25_only'
        - 'dense_only'
        - 'hybrid'
        - 'hybrid_rerank'
        """
        # Scaffolding pipeline execution - to be wired to indexes in next milestone
        return make_refusal_response(
            reason="Pipeline index not yet populated",
            retrieved_chunk_ids=[],
        )
