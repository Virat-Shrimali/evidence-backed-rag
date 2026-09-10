"""Lightweight HTTP API client connecting the Streamlit frontend to the FastAPI backend."""

from __future__ import annotations

import os
from typing import Any

import httpx

DEFAULT_BACKEND_URL = "http://localhost:8000"
PRODUCTION_BACKEND_URL = "https://evidence-backed-rag.onrender.com"


class BackendClientError(Exception):
    """User-friendly exception for backend API errors, sanitizing raw exceptions."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


def get_backend_url(default: str | None = None) -> str:
    """Retrieve backend URL from Streamlit secrets, environment variables, or fallback.

    Resolution order:
    1. streamlit.secrets["BACKEND_URL"] (if running inside Streamlit with secrets configured)
    2. os.environ["BACKEND_URL"]
    3. Provided default parameter
    4. DEFAULT_BACKEND_URL ("http://localhost:8000")
    """
    # 1. Streamlit secrets
    try:
        import streamlit as st

        if hasattr(st, "secrets") and "BACKEND_URL" in st.secrets:
            url = str(st.secrets["BACKEND_URL"]).strip()
            if url:
                return url.rstrip("/")
    except Exception:
        pass

    # 2. Environment variable
    env_url = os.environ.get("BACKEND_URL", "").strip()
    if env_url:
        return env_url.rstrip("/")

    # 3. Fallback
    return (default or DEFAULT_BACKEND_URL).rstrip("/")


def build_query_payload(
    question: str,
    retriever_mode: str = "bm25_only",
    top_k: int = 5,
) -> dict[str, Any]:
    """Construct JSON payload for the /query endpoint."""
    return {
        "question": question.strip(),
        "retriever_mode": retriever_mode,
        "top_k": top_k,
    }


def parse_backend_response(data: Any) -> dict[str, Any]:
    """Validate and normalize backend JSON response into structured response dictionary."""
    if not isinstance(data, dict):
        raise BackendClientError("Backend returned an invalid, non-dictionary JSON response.")

    if "answer" not in data:
        raise BackendClientError("Backend response is missing required 'answer' field.")

    citations = data.get("citations", [])
    if not isinstance(citations, list):
        citations = []

    normalized_citations: list[dict[str, str]] = []
    for c in citations:
        if isinstance(c, dict):
            cid = str(c.get("chunk_id", "")).strip()
            snippet = str(c.get("text_snippet", c.get("snippet", ""))).strip()
            normalized_citations.append({"chunk_id": cid, "text_snippet": snippet})

    return {
        "answer": str(data.get("answer", "")).strip(),
        "citations": normalized_citations,
        "sufficient_evidence": bool(data.get("sufficient_evidence", True)),
        "confidence": float(data.get("confidence", 1.0 if data.get("sufficient_evidence") else 0.0)),
        "retrieved_chunk_ids": [str(x) for x in data.get("retrieved_chunk_ids", [])],
        "refusal_reason": data.get("refusal_reason"),
    }


def query_backend(
    question: str,
    retriever_mode: str = "bm25_only",
    top_k: int = 5,
    backend_url: str | None = None,
    timeout: float = 30.0,
    client: httpx.Client | None = None,
) -> dict[str, Any]:
    """Execute evidence-backed query against the FastAPI backend.

    Sends POST {backend_url}/query and returns parsed response.
    Catches timeouts, connection errors, and HTTP status codes, raising
    clean, user-facing BackendClientError exceptions without exposing stack traces.
    """
    base_url = (backend_url or get_backend_url()).rstrip("/")
    endpoint = f"{base_url}/query"
    payload = build_query_payload(question=question, retriever_mode=retriever_mode, top_k=top_k)

    headers = {"Content-Type": "application/json", "Accept": "application/json"}

    def _do_request(http_client: httpx.Client) -> httpx.Response:
        return http_client.post(endpoint, json=payload, headers=headers, timeout=timeout)

    try:
        if client is not None:
            response = _do_request(client)
        else:
            with httpx.Client(timeout=timeout) as managed_client:
                response = _do_request(managed_client)
    except httpx.TimeoutException as exc:
        raise BackendClientError(
            f"Request to backend timed out after {int(timeout)}s. "
            "If the backend service is waking from cold start, please wait 30 seconds and retry."
        ) from exc
    except httpx.NetworkError as exc:
        raise BackendClientError(
            f"Unable to connect to backend at {base_url}. "
            "Please check network connectivity or verify that the backend service is running."
        ) from exc
    except Exception as exc:
        raise BackendClientError(
            "An unexpected error occurred while connecting to the backend service."
        ) from exc

    # Handle HTTP status errors
    if response.status_code == 404:
        raise BackendClientError("Backend query endpoint not found (HTTP 404).", status_code=404)
    if response.status_code == 422:
        raise BackendClientError(
            "Backend rejected the query request as unprocessable (HTTP 422).", status_code=422
        )
    if response.status_code in (502, 503, 504):
        raise BackendClientError(
            f"Backend service is temporarily unavailable (HTTP {response.status_code}). "
            "The service may be waking up or restarting; please wait a moment and try again.",
            status_code=response.status_code,
        )
    if response.status_code >= 500:
        raise BackendClientError(
            f"Backend encountered an internal error (HTTP {response.status_code}).",
            status_code=response.status_code,
        )
    if response.status_code != 200:
        raise BackendClientError(
            f"Backend returned unexpected HTTP status {response.status_code}.",
            status_code=response.status_code,
        )

    # Parse JSON
    try:
        raw_json = response.json()
    except Exception as exc:
        raise BackendClientError("Backend returned an invalid, non-JSON response body.") from exc

    return parse_backend_response(raw_json)
