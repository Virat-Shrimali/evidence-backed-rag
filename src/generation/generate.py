"""Generation execution, schema validation, and refusal enforcement."""

import json

from pydantic import BaseModel, Field

from src.config import INSUFFICIENT_EVIDENCE_REFUSAL, settings


class Citation(BaseModel):
    """Verifiable chunk citation."""
    chunk_id: str = Field(description="Exact ID of the cited chunk")
    snippet: str = Field(description="Exact quotation or factual snippet from chunk")


class RAGResponse(BaseModel):
    """Structured response output for RAG query."""
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    sufficient_evidence: bool = True
    confidence: float = 1.0
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    refusal_reason: str | None = None


def make_refusal_response(
    reason: str = "Evidence confidence below threshold",
    retrieved_chunk_ids: list[str] | None = None,
) -> RAGResponse:
    """Create a standardized refusal response."""
    return RAGResponse(
        answer=INSUFFICIENT_EVIDENCE_REFUSAL,
        citations=[],
        sufficient_evidence=False,
        confidence=0.0,
        retrieved_chunk_ids=retrieved_chunk_ids or [],
        refusal_reason=reason,
    )


def parse_llm_json_response(raw_text: str, retrieved_chunk_ids: list[str]) -> RAGResponse:
    """Safely parse LLM JSON output into a validated RAGResponse."""
    try:
        cleaned = raw_text.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        data = json.loads(cleaned)

        sufficient = data.get("sufficient_evidence", True)
        confidence = float(data.get("confidence", 1.0 if sufficient else 0.0))

        if not sufficient or confidence < settings.evidence_confidence_threshold:
            return make_refusal_response(
                reason=data.get("refusal_reason", "Insufficient context found"),
                retrieved_chunk_ids=retrieved_chunk_ids,
            )

        citations_data = data.get("citations", [])
        citations = [Citation(**c) for c in citations_data]

        return RAGResponse(
            answer=data.get("answer", ""),
            citations=citations,
            sufficient_evidence=True,
            confidence=confidence,
            retrieved_chunk_ids=retrieved_chunk_ids,
            refusal_reason=None,
        )
    except Exception as exc:
        return make_refusal_response(
            reason=f"Failed to parse generation output: {exc}",
            retrieved_chunk_ids=retrieved_chunk_ids,
        )
