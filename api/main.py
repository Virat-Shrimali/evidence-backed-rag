"""FastAPI production service exposing evidence-backed RAG endpoints."""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from src.generation.generate import RAGResponse
from src.pipeline import RAGPipeline

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Evidence-Backed RAG API",
    version="0.1.0",
    description="Production Question Answering API with verifiable chunk citations and deterministic refusal.",
)

_pipeline_instance: RAGPipeline | None = None


def get_pipeline() -> RAGPipeline:
    """Dependency provider for RAGPipeline instance."""
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = RAGPipeline()
    return _pipeline_instance


class QueryRequest(BaseModel):
    """Payload for POST /query endpoint."""

    question: str = Field(..., description="User question to answer against document evidence.")
    retriever_mode: str | None = Field(
        default=None,
        description="Optional retriever strategy override ('bm25_only', 'dense_only', 'hybrid', 'hybrid_rerank').",
    )
    top_k: int | None = Field(
        default=None,
        ge=1,
        le=50,
        description="Optional number of evidence chunks to retrieve.",
    )

    @field_validator("question")
    @classmethod
    def validate_non_empty(cls, v: str) -> str:
        trimmed = v.strip()
        if not trimmed:
            raise ValueError("Question must not be empty or whitespace only.")
        return trimmed


class HealthResponse(BaseModel):
    """Payload for GET /health endpoint."""

    status: str = "healthy"
    service: str = "evidence-backed-rag"


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Safely catch unhandled exceptions without leaking internal trace details."""
    logger.error(
        "Unhandled error processing request %s %s: %s",
        request.method,
        request.url.path,
        exc,
        exc_info=True,
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An internal server error occurred while processing the request."},
    )


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Health check endpoint confirming operational readiness."""
    return HealthResponse(status="healthy", service="evidence-backed-rag")


@app.post("/query", response_model=RAGResponse, status_code=status.HTTP_200_OK)
def query_rag(
    request: QueryRequest,
    pipeline: Annotated[RAGPipeline, Depends(get_pipeline)],
) -> RAGResponse:
    """Execute evidence-backed RAG query with strict citation grounding and refusal."""
    logger.info(
        "Received query request: '%s' (retriever_mode=%s, top_k=%s)",
        request.question,
        request.retriever_mode,
        request.top_k,
    )
    try:
        return pipeline.query(
            question=request.question,
            retriever_mode=request.retriever_mode,
            top_k=request.top_k,
        )
    except Exception as exc:
        logger.error(
            "Error executing RAG pipeline for query '%s': %s",
            request.question,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while executing the RAG pipeline.",
        ) from exc

