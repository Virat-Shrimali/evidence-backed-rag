"""Tests for Streamlit UI helpers, API client, formatting utilities, and import integrity."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from unittest.mock import MagicMock

import httpx
import pytest

from app.api_client import (
    DEFAULT_BACKEND_URL,
    BackendClientError,
    build_query_payload,
    get_backend_url,
    parse_backend_response,
    query_backend,
)
from app.streamlit_app import (
    SAMPLE_QUESTIONS,
    STRATEGY_OPTIONS,
    format_strategy_name,
    get_selectable_strategies,
    parse_chunk_provenance,
)


def test_streamlit_app_imports_cleanly():
    """Verify that app.streamlit_app can be imported cleanly without top-level errors."""
    module = importlib.import_module("app.streamlit_app")
    assert hasattr(module, "main"), "streamlit_app must expose main()"
    assert hasattr(module, "STRATEGY_OPTIONS"), "streamlit_app must define STRATEGY_OPTIONS"
    assert hasattr(module, "parse_chunk_provenance"), "streamlit_app must define parse_chunk_provenance"
    assert hasattr(module, "get_selectable_strategies"), "streamlit_app must define get_selectable_strategies"


def test_api_client_imports_cleanly():
    """Verify that app.api_client can be imported cleanly without errors."""
    module = importlib.import_module("app.api_client")
    assert hasattr(module, "query_backend"), "api_client must expose query_backend"
    assert hasattr(module, "get_backend_url"), "api_client must expose get_backend_url"
    assert hasattr(module, "BackendClientError"), "api_client must expose BackendClientError"


def test_frontend_does_not_instantiate_rag_pipeline():
    """Verify frontend modules do not instantiate or require RAGPipeline."""
    import app.api_client as client_mod
    import app.streamlit_app as ui_mod

    assert not hasattr(ui_mod, "RAGPipeline"), "Streamlit frontend must not import or instantiate RAGPipeline"
    assert not hasattr(client_mod, "RAGPipeline"), "API client must not import or instantiate RAGPipeline"


def test_frontend_does_not_import_heavy_ml_libraries():
    """Verify frontend and client modules do not import torch, sentence_transformers, or chromadb."""
    import app.api_client as client_mod
    import app.streamlit_app as ui_mod

    for heavy_mod in ("torch", "sentence_transformers", "chromadb", "transformers"):
        assert not hasattr(ui_mod, heavy_mod), f"streamlit_app must not import {heavy_mod}"
        assert not hasattr(client_mod, heavy_mod), f"api_client must not import {heavy_mod}"


def test_app_import_fallback_when_app_dir_in_syspath():
    """Verify app/streamlit_app.py imports api_client robustly when executed directly in Streamlit."""
    app_dir = str(Path(__file__).resolve().parent.parent / "app")

    # Temporarily prepend app directory to sys.path to simulate Streamlit Community Cloud runner
    original_path = list(sys.path)
    try:
        sys.path.insert(0, app_dir)
        # Re-import to confirm robust resolution
        import app.streamlit_app as re_ui_mod

        assert hasattr(re_ui_mod, "main")
    finally:
        sys.path = original_path


def test_strategy_options_completeness():
    """Verify that all four required retrieval strategies are available in UI mappings."""
    expected_keys = {"hybrid_rerank", "hybrid", "dense_only", "bm25_only"}
    actual_keys = set(STRATEGY_OPTIONS.values())
    assert expected_keys.issubset(actual_keys), f"Missing strategies: {expected_keys - actual_keys}"

    for label, key in STRATEGY_OPTIONS.items():
        assert format_strategy_name(key) == label


def test_selectable_strategies_for_render_free():
    """Verify strategy selector restricts to bm25_only when targeting Render Free backend."""
    render_url = "https://evidence-backed-rag.onrender.com"
    strategies = get_selectable_strategies(render_url)
    assert len(strategies) == 1
    assert list(strategies.values()) == ["bm25_only"]
    assert "Render Free Safe" in list(strategies.keys())[0]


def test_selectable_strategies_for_local_unconstrained():
    """Verify strategy selector exposes all strategies for local/high-memory backend."""
    local_url = "http://localhost:8000"
    strategies = get_selectable_strategies(local_url)
    assert len(strategies) == 4
    assert set(strategies.values()) == {"hybrid_rerank", "hybrid", "dense_only", "bm25_only"}


def test_parse_chunk_provenance_strategy_a():
    """Verify parsing of Strategy A deterministic chunk ID."""
    chunk_id = "doc_evidence-backed-rag-project-guide_1a754e0a#stratA#c0003"
    info = parse_chunk_provenance(chunk_id)

    assert info["document_id"] == "doc_evidence-backed-rag-project-guide_1a754e0a"
    assert info["document_name"] == "evidence-backed-rag-project-guide"
    assert info["strategy"] == "Strategy A (Fixed 500)"
    assert info["chunk_index"] == "c0003"


def test_parse_chunk_provenance_strategy_b():
    """Verify parsing of Strategy B deterministic chunk ID."""
    chunk_id = "doc_financial-report-2024_deadbeef#stratB#c0012"
    info = parse_chunk_provenance(chunk_id)

    assert info["document_id"] == "doc_financial-report-2024_deadbeef"
    assert info["document_name"] == "financial-report-2024"
    assert info["strategy"] == "Strategy B (Sentence 200)"
    assert info["chunk_index"] == "c0012"


def test_parse_chunk_provenance_fallback():
    """Verify fallback behavior for plain or non-conforming chunk IDs."""
    info = parse_chunk_provenance("plain_chunk_id")
    assert info["document_id"] == "plain_chunk_id"
    assert info["strategy"] == "unknown"
    assert info["chunk_index"] == ""


def test_sample_questions_include_refusal_test():
    """Verify that sample questions include an unanswerable query for refusal testing."""
    assert len(SAMPLE_QUESTIONS) >= 4
    has_refusal = any("PostgreSQL" in q or "Martian" in q or "recipe" in q for q in SAMPLE_QUESTIONS)
    assert has_refusal, "Sample queries should include an unanswerable query for refusal demonstration."


# ==============================================================================
# API Client Tests
# ==============================================================================


def test_get_backend_url_resolution(monkeypatch):
    """Verify backend URL resolution across environment variables and fallbacks."""
    # 1. Fallback default
    monkeypatch.delenv("BACKEND_URL", raising=False)
    assert get_backend_url() == DEFAULT_BACKEND_URL

    # 2. Provided explicit default
    assert get_backend_url(default="http://custom-host:9000") == "http://custom-host:9000"

    # 3. Environment variable override
    monkeypatch.setenv("BACKEND_URL", "https://evidence-backed-rag.onrender.com/")
    assert get_backend_url() == "https://evidence-backed-rag.onrender.com"


def test_get_backend_url_with_streamlit_secrets(monkeypatch):
    """Verify get_backend_url reads from st.secrets if available."""
    mock_st = MagicMock()
    mock_st.secrets = {"BACKEND_URL": "https://secrets-backend.example.com"}
    monkeypatch.setitem(sys.modules, "streamlit", mock_st)

    assert get_backend_url() == "https://secrets-backend.example.com"


def test_build_query_payload():
    """Verify JSON request payload construction."""
    payload = build_query_payload("What is RAG?", retriever_mode="bm25_only", top_k=7)
    assert payload == {
        "question": "What is RAG?",
        "retriever_mode": "bm25_only",
        "top_k": 7,
    }


def test_parse_backend_response_success():
    """Verify parsing of valid backend response with citations."""
    raw_data = {
        "answer": "Operating margin increased by 3.5%.",
        "citations": [
            {"chunk_id": "doc1#stratA#c0001", "text_snippet": "Operating margin increased by 3.5%"},
            {"chunk_id": "doc1#stratA#c0002", "snippet": "revenue reached $120M"},
        ],
        "sufficient_evidence": True,
        "confidence": 0.95,
        "retrieved_chunk_ids": ["doc1#stratA#c0001", "doc1#stratA#c0002"],
        "refusal_reason": None,
    }
    parsed = parse_backend_response(raw_data)
    assert parsed["answer"] == "Operating margin increased by 3.5%."
    assert parsed["sufficient_evidence"] is True
    assert parsed["confidence"] == 0.95
    assert len(parsed["citations"]) == 2
    assert parsed["citations"][0]["chunk_id"] == "doc1#stratA#c0001"
    assert parsed["citations"][0]["text_snippet"] == "Operating margin increased by 3.5%"
    assert parsed["citations"][1]["chunk_id"] == "doc1#stratA#c0002"
    assert parsed["citations"][1]["text_snippet"] == "revenue reached $120M"
    assert parsed["retrieved_chunk_ids"] == ["doc1#stratA#c0001", "doc1#stratA#c0002"]
    assert parsed["refusal_reason"] is None


def test_parse_backend_response_refusal():
    """Verify parsing of deterministic refusal response."""
    raw_data = {
        "answer": "Insufficient evidence to answer this question based on the provided documents.",
        "citations": [],
        "sufficient_evidence": False,
        "confidence": 0.0,
        "retrieved_chunk_ids": ["doc1#c0001"],
        "refusal_reason": "No evidence found in context",
    }
    parsed = parse_backend_response(raw_data)
    assert parsed["sufficient_evidence"] is False
    assert "Insufficient evidence" in parsed["answer"]
    assert parsed["confidence"] == 0.0
    assert parsed["citations"] == []
    assert parsed["refusal_reason"] == "No evidence found in context"


def test_parse_backend_response_invalid_structure():
    """Verify parsing rejects malformed/non-dictionary payloads."""
    with pytest.raises(BackendClientError, match="invalid, non-dictionary"):
        parse_backend_response(["not", "a", "dict"])

    with pytest.raises(BackendClientError, match="missing required 'answer'"):
        parse_backend_response({"no_answer": True})


def test_query_backend_successful_mock():
    """Verify query_backend executes POST /query and returns parsed response."""
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "answer": "MiniLM is used.",
        "citations": [{"chunk_id": "c8", "text_snippet": "MiniLM"}],
        "sufficient_evidence": True,
        "confidence": 0.9,
        "retrieved_chunk_ids": ["c8"],
        "refusal_reason": None,
    }

    mock_client = MagicMock(spec=httpx.Client)
    mock_client.post.return_value = mock_resp

    result = query_backend(
        question="What model is used?",
        backend_url="http://mock-backend:8000",
        client=mock_client,
    )

    assert result["answer"] == "MiniLM is used."
    assert result["sufficient_evidence"] is True
    assert len(result["citations"]) == 1
    mock_client.post.assert_called_once()
    args, kwargs = mock_client.post.call_args
    assert args[0] == "http://mock-backend:8000/query"
    assert kwargs["json"] == {
        "question": "What model is used?",
        "retriever_mode": "bm25_only",
        "top_k": 5,
    }


def test_query_backend_timeout_error():
    """Verify query_backend handles timeout gracefully."""
    mock_client = MagicMock(spec=httpx.Client)
    mock_client.post.side_effect = httpx.TimeoutException("Read timed out")

    with pytest.raises(BackendClientError, match="timed out"):
        query_backend("Test query", client=mock_client)


def test_query_backend_network_error():
    """Verify query_backend handles network/connection failure gracefully."""
    mock_client = MagicMock(spec=httpx.Client)
    mock_client.post.side_effect = httpx.ConnectError("Failed to connect")

    with pytest.raises(BackendClientError, match="Unable to connect to backend"):
        query_backend("Test query", client=mock_client)


@pytest.mark.parametrize(
    "status_code,err_match",
    [
        (404, "endpoint not found"),
        (422, "rejected the query request"),
        (502, "temporarily unavailable"),
        (500, "internal error"),
    ],
)
def test_query_backend_http_errors(status_code, err_match):
    """Verify query_backend translates HTTP status errors into user-friendly messages."""
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = status_code
    mock_client = MagicMock(spec=httpx.Client)
    mock_client.post.return_value = mock_resp

    with pytest.raises(BackendClientError, match=err_match) as exc_info:
        query_backend("Test query", client=mock_client)
    assert exc_info.value.status_code == status_code


def test_query_backend_malformed_json():
    """Verify query_backend handles non-JSON response body."""
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 200
    mock_resp.json.side_effect = ValueError("Invalid JSON")
    mock_client = MagicMock(spec=httpx.Client)
    mock_client.post.return_value = mock_resp

    with pytest.raises(BackendClientError, match="invalid, non-JSON"):
        query_backend("Test query", client=mock_client)
