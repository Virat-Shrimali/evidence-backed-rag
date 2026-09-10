"""Tests for Streamlit UI helpers, formatting utilities, and import integrity."""

from __future__ import annotations

import importlib


def test_streamlit_app_imports_cleanly():
    """Verify that app.streamlit_app can be imported cleanly without top-level errors."""
    module = importlib.import_module("app.streamlit_app")
    assert hasattr(module, "main"), "streamlit_app must expose main()"
    assert hasattr(module, "STRATEGY_OPTIONS"), "streamlit_app must define STRATEGY_OPTIONS"
    assert hasattr(module, "parse_chunk_provenance"), "streamlit_app must define parse_chunk_provenance"


def test_strategy_options_completeness():
    """Verify that all four required retrieval strategies are available in UI mappings."""
    from app.streamlit_app import STRATEGY_OPTIONS, format_strategy_name

    expected_keys = {"hybrid_rerank", "hybrid", "dense_only", "bm25_only"}
    actual_keys = set(STRATEGY_OPTIONS.values())
    assert expected_keys.issubset(actual_keys), f"Missing strategies: {expected_keys - actual_keys}"

    for label, key in STRATEGY_OPTIONS.items():
        assert format_strategy_name(key) == label


def test_parse_chunk_provenance_strategy_a():
    """Verify parsing of Strategy A deterministic chunk ID."""
    from app.streamlit_app import parse_chunk_provenance

    chunk_id = "doc_evidence-backed-rag-project-guide_1a754e0a#stratA#c0003"
    info = parse_chunk_provenance(chunk_id)

    assert info["document_id"] == "doc_evidence-backed-rag-project-guide_1a754e0a"
    assert info["document_name"] == "evidence-backed-rag-project-guide"
    assert info["strategy"] == "Strategy A (Fixed 500)"
    assert info["chunk_index"] == "c0003"


def test_parse_chunk_provenance_strategy_b():
    """Verify parsing of Strategy B deterministic chunk ID."""
    from app.streamlit_app import parse_chunk_provenance

    chunk_id = "doc_financial-report-2024_deadbeef#stratB#c0012"
    info = parse_chunk_provenance(chunk_id)

    assert info["document_id"] == "doc_financial-report-2024_deadbeef"
    assert info["document_name"] == "financial-report-2024"
    assert info["strategy"] == "Strategy B (Sentence 200)"
    assert info["chunk_index"] == "c0012"


def test_parse_chunk_provenance_fallback():
    """Verify fallback behavior for plain or non-conforming chunk IDs."""
    from app.streamlit_app import parse_chunk_provenance

    info = parse_chunk_provenance("plain_chunk_id")
    assert info["document_id"] == "plain_chunk_id"
    assert info["strategy"] == "unknown"
    assert info["chunk_index"] == ""


def test_sample_questions_include_refusal_test():
    """Verify that sample questions include an unanswerable query for refusal testing."""
    from app.streamlit_app import SAMPLE_QUESTIONS

    assert len(SAMPLE_QUESTIONS) >= 4
    # At least one question tests the refusal path
    has_refusal = any("Martian" in q or "recipe" in q for q in SAMPLE_QUESTIONS)
    assert has_refusal, "Sample queries should include an unanswerable query for refusal demonstration."
