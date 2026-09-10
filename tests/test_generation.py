"""Unit tests for generation parsing, citations, and refusal enforcement."""

from src.config import INSUFFICIENT_EVIDENCE_REFUSAL
from src.generation.generate import parse_llm_json_response


def test_parse_valid_llm_response():
    """Verify structured response with citations is correctly parsed."""
    raw_json = """
    {
        "answer": "Revenue reached 120 million in FY2023.",
        "citations": [{"chunk_id": "doc1#chunk-0", "snippet": "revenue of $120 million"}],
        "sufficient_evidence": true,
        "confidence": 0.95
    }
    """
    response = parse_llm_json_response(raw_json, retrieved_chunk_ids=["doc1#chunk-0"])
    assert response.sufficient_evidence is True
    assert response.confidence == 0.95
    assert len(response.citations) == 1
    assert response.citations[0].chunk_id == "doc1#chunk-0"
    assert response.citations[0].snippet == "revenue of $120 million"
    assert response.refusal_reason is None


def test_refusal_on_insufficient_evidence():
    """Verify system strictly returns standard refusal message when evidence is insufficient."""
    raw_json = """
    {
        "answer": "I am not sure about the quarterly dividend rate.",
        "citations": [],
        "sufficient_evidence": false,
        "confidence": 0.1,
        "refusal_reason": "No dividend information mentioned"
    }
    """
    response = parse_llm_json_response(raw_json, retrieved_chunk_ids=["doc1#chunk-0"])
    assert response.sufficient_evidence is False
    assert response.answer == INSUFFICIENT_EVIDENCE_REFUSAL
    assert response.confidence == 0.0


def test_refusal_on_low_confidence():
    """Verify refusal triggers if confidence is below threshold even if marked sufficient."""
    raw_json = """
    {
        "answer": "Maybe it was 10 percent.",
        "citations": [],
        "sufficient_evidence": true,
        "confidence": 0.40
    }
    """
    response = parse_llm_json_response(raw_json, retrieved_chunk_ids=["doc1#chunk-0"])
    assert response.sufficient_evidence is False
    assert response.answer == INSUFFICIENT_EVIDENCE_REFUSAL


def test_refusal_on_malformed_json():
    """Verify malformed JSON gracefully triggers standard refusal instead of crashing."""
    malformed = "This is not valid JSON at all."
    response = parse_llm_json_response(malformed, retrieved_chunk_ids=["doc1#chunk-0"])
    assert response.sufficient_evidence is False
    assert response.answer == INSUFFICIENT_EVIDENCE_REFUSAL
