"""Generation execution, schema validation, and deterministic refusal enforcement."""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from typing import TYPE_CHECKING, Any

from pydantic import AliasChoices, BaseModel, Field

from src.config import INSUFFICIENT_EVIDENCE_REFUSAL, RAGConfig, settings
from src.generation.prompt_templates import (
    STRICT_RAG_SYSTEM_PROMPT,
    format_context_prompt,
)
from src.generation.providers import BaseLLMProvider, create_llm_provider

if TYPE_CHECKING:
    from src.retrieval.models import RetrievedChunk

logger = logging.getLogger(__name__)


class Citation(BaseModel):
    """Verifiable chunk citation."""

    chunk_id: str = Field(description="Exact ID of the cited chunk")
    text_snippet: str = Field(
        default="",
        description="Verbatim text quotation or factual snippet extracted from chunk",
        validation_alias=AliasChoices("text_snippet", "snippet"),
    )

    @property
    def snippet(self) -> str:
        """Backwards-compatible accessor for snippet text."""
        return self.text_snippet


class RAGResponse(BaseModel):
    """Structured response output for RAG query."""

    answer: str
    citations: list[Citation] = Field(default_factory=list)
    sufficient_evidence: bool = True
    confidence: float = Field(
        default=1.0,
        description=(
            "Heuristic grounding confidence score (0.0 to 1.0) indicating how strongly "
            "the generated answer is supported by retrieved evidence chunks and whether "
            "evidence passed validation thresholds. This is a heuristic grounding indicator, "
            "NOT a calibrated statistical probability."
        ),
    )
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


def normalize_for_matching(text: str) -> str:
    """Normalize text by NFKC normalizing, lowercasing, and collapsing whitespace."""
    if not text:
        return ""
    norm = unicodedata.normalize("NFKC", text).lower()
    return re.sub(r"\s+", " ", norm).strip()


def validate_citations(
    citations: list[Citation],
    retrieved_chunks: list[Any] | None,
    retrieved_chunk_ids: list[str] | None = None,
) -> tuple[bool, list[Citation], str | None]:
    """Deterministically validate citations against the retrieved chunks.

    Enforces:
    1. Every cited chunk_id must exist in the retrieved context.
    2. Citation text snippets must be grounded in the cited chunk's content (if full chunk objects provided).
    3. Deduplicates identical citations.
    4. Never silently fabricates citations.
    """
    if not citations:
        return True, [], None

    # Determine allowable chunk IDs and mapping to chunk content
    allowable_ids: set[str] = set(retrieved_chunk_ids or [])
    chunk_content_map: dict[str, str] = {}

    if retrieved_chunks:
        for chunk in retrieved_chunks:
            cid = getattr(chunk, "chunk_id", getattr(chunk, "id", None))
            if cid:
                allowable_ids.add(cid)
                chunk_content_map[cid] = getattr(chunk, "content", "")

    validated_citations: list[Citation] = []
    seen_citations: set[tuple[str, str]] = set()

    for citation in citations:
        cid = citation.chunk_id.strip()
        # 1. Non-existent chunk ID check
        if cid not in allowable_ids:
            return (
                False,
                [],
                f"Citation referenced non-existent chunk ID: '{cid}' not found in retrieved chunks",
            )

        # 2. Grounding snippet check (if chunk content available)
        raw_snippet = citation.text_snippet.strip()
        if raw_snippet and cid in chunk_content_map:
            chunk_content = chunk_content_map[cid]
            norm_snippet = normalize_for_matching(raw_snippet)
            norm_chunk = normalize_for_matching(chunk_content)

            # Check if snippet appears in chunk content
            if norm_snippet not in norm_chunk:
                # Also allow prefix match for long snippets
                prefix_len = min(40, len(norm_snippet))
                if norm_snippet[:prefix_len] not in norm_chunk:
                    return (
                        False,
                        [],
                        f"Citation snippet '{raw_snippet[:50]}...' was not found in cited chunk '{cid}'",
                    )

        # 3. Deduplicate
        citation_key = (cid, raw_snippet)
        if citation_key not in seen_citations:
            seen_citations.add(citation_key)
            validated_citations.append(citation)

    return True, validated_citations, None


def parse_llm_json_response(
    raw_text: str,
    retrieved_chunk_ids: list[str],
    retrieved_chunks: list[Any] | None = None,
    confidence_threshold: float | None = None,
) -> RAGResponse:
    """Safely parse LLM JSON output into a validated RAGResponse."""
    thresh = (
        confidence_threshold
        if confidence_threshold is not None
        else settings.evidence_confidence_threshold
    )
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
        answer = str(data.get("answer", "")).strip()

        # Deterministic check: explicit refusal from LLM or below confidence threshold
        if (
            not sufficient
            or confidence < thresh
            or answer == INSUFFICIENT_EVIDENCE_REFUSAL
        ):
            return make_refusal_response(
                reason=data.get("refusal_reason", "Insufficient context found"),
                retrieved_chunk_ids=retrieved_chunk_ids,
            )

        citations_data = data.get("citations", [])
        raw_citations = [Citation(**c) for c in citations_data]

        # Deterministic evidence validation: an answer marked sufficient MUST have citations
        if not raw_citations:
            return make_refusal_response(
                reason="Answer claimed sufficient evidence but provided no citations",
                retrieved_chunk_ids=retrieved_chunk_ids,
            )

        # Validate that cited chunks exist and snippets are grounded
        is_valid, validated_citations, val_err = validate_citations(
            raw_citations,
            retrieved_chunks=retrieved_chunks,
            retrieved_chunk_ids=retrieved_chunk_ids,
        )
        if not is_valid:
            return make_refusal_response(
                reason=val_err or "Citation validation failed",
                retrieved_chunk_ids=retrieved_chunk_ids,
            )

        return RAGResponse(
            answer=answer,
            citations=validated_citations,
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


class EvidenceGroundedGenerator:
    """Provider-agnostic generator producing grounded answers with verifiable citations."""

    def __init__(
        self,
        provider: BaseLLMProvider | None = None,
        config: RAGConfig | None = None,
        confidence_threshold: float | None = None,
    ):
        self.config = config or settings
        self.provider = provider or create_llm_provider(config=self.config)
        self.confidence_threshold = (
            confidence_threshold
            if confidence_threshold is not None
            else self.config.evidence_confidence_threshold
        )

    def generate(
        self,
        query: str,
        retrieved_chunks: list[RetrievedChunk] | list[Any],
    ) -> RAGResponse:
        """Generate an evidence-grounded response adhering to strict citation rules.

        - Rejects queries with empty retrieved chunks (deterministic refusal).
        - Rejects queries where retrieval scores are below confidence threshold.
        - Validates all citations and text snippets against retrieved chunks.
        - Returns structured RAGResponse.
        """
        # Extract chunk IDs
        retrieved_chunk_ids: list[str] = []
        for c in retrieved_chunks:
            cid = getattr(c, "chunk_id", getattr(c, "id", None))
            if cid:
                retrieved_chunk_ids.append(cid)

        # 1. Deterministic refusal on empty evidence
        if not retrieved_chunks or not query.strip():
            return make_refusal_response(
                reason="No retrieved context chunks provided" if not retrieved_chunks else "Empty query string",
                retrieved_chunk_ids=retrieved_chunk_ids,
            )

        # 2. Deterministic refusal on low retrieval score (if scores exist)
        # Check top chunk's score against threshold if rerank_score or score is present
        top_chunk = retrieved_chunks[0]
        if hasattr(top_chunk, "rerank_score") and top_chunk.rerank_score is not None:
            # Neural reranker logits: threshold typically 0.0 or negative for non-relevant
            if top_chunk.rerank_score < -2.0:
                return make_refusal_response(
                    reason=f"Top candidate rerank score {top_chunk.rerank_score:.3f} below confidence threshold",
                    retrieved_chunk_ids=retrieved_chunk_ids,
                )
        elif hasattr(top_chunk, "score") and top_chunk.score is not None:
            # Cosine similarity for dense: check against evidence_confidence_threshold (e.g. 0.65)
            # Only apply if method is dense
            if getattr(top_chunk, "retrieval_method", "") == "dense":
                if top_chunk.score < self.confidence_threshold:
                    return make_refusal_response(
                        reason=f"Top candidate similarity {top_chunk.score:.3f} below threshold {self.confidence_threshold}",
                        retrieved_chunk_ids=retrieved_chunk_ids,
                    )

        # 3. Format prompt with strict system instructions and context
        prompt = format_context_prompt(retrieved_chunks, question=query)

        # 4. Generate raw output from provider
        try:
            raw_output = self.provider.generate_raw(
                prompt=prompt,
                system_prompt=STRICT_RAG_SYSTEM_PROMPT,
            )
        except Exception as e:
            logger.error("Provider generation error: %s", e)
            return make_refusal_response(
                reason=f"Provider generation failure: {e}",
                retrieved_chunk_ids=retrieved_chunk_ids,
            )

        # 5. Parse, validate, and return structured output
        return parse_llm_json_response(
            raw_text=raw_output,
            retrieved_chunk_ids=retrieved_chunk_ids,
            retrieved_chunks=retrieved_chunks,
            confidence_threshold=self.confidence_threshold,
        )

