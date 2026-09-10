"""FastAPI backend service exposing RAG endpoints."""


from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.generation.generate import RAGResponse
from src.pipeline import RAGPipeline

app = FastAPI(
    title="Evidence-Backed RAG API",
    version="0.1.0",
    description="Q&A API with chunk citations and refusal logic.",
)

pipeline = RAGPipeline()


class QueryRequest(BaseModel):
    """Payload for POST /query endpoint."""
    question: str = Field(..., min_length=1, description="User question")
    retriever_mode: str | None = Field(
        default="hybrid_rerank",
        description="Retriever variant: 'bm25_only', 'dense_only', 'hybrid', 'hybrid_rerank'",
    )


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "evidence-backed-rag"}


@app.post("/query", response_model=RAGResponse)
def query_rag(request: QueryRequest) -> RAGResponse:
    """Execute RAG query against document corpus."""
    return pipeline.query(
        question=request.question,
        retriever_mode=request.retriever_mode or "hybrid_rerank",
    )
