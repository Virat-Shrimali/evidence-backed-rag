"""Unit tests for FastAPI production endpoints."""

import pytest
from fastapi.testclient import TestClient

from api.main import app, get_pipeline
from src.config import INSUFFICIENT_EVIDENCE_REFUSAL
from src.generation.generate import EvidenceGroundedGenerator, RAGResponse
from src.generation.providers import MockLLMProvider
from src.pipeline import RAGPipeline
from src.retrieval.models import RetrievedChunk
from src.retrieval.retriever import BaseRetriever


class MockAPIRetriever(BaseRetriever):
    """Mock retriever returning deterministic candidates for API testing."""

    def __init__(self, chunks: list[RetrievedChunk] | None = None):
        self.chunks = chunks or []

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        if "unanswerable" in query.lower() or "missing" in query.lower():
            return []
        return self.chunks[:top_k] if top_k else self.chunks


@pytest.fixture
def mock_api_chunks() -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            chunk_id="doc1#stratA#c0001",
            document_id="doc1",
            content="FastAPI is an asynchronous Python framework with automatic documentation.",
            score=0.92,
            rank=1,
            retrieval_method="dense",
            page_numbers=[1],
        )
    ]


@pytest.fixture
def api_client(mock_api_chunks: list[RetrievedChunk]) -> TestClient:
    """Create a TestClient with an isolated, offline mock pipeline."""
    canned_llm_response = """
    {
        "answer": "FastAPI is an asynchronous Python framework.",
        "citations": [{"chunk_id": "doc1#stratA#c0001", "text_snippet": "FastAPI is an asynchronous Python framework"}],
        "sufficient_evidence": true,
        "confidence": 0.98
    }
    """
    mock_retriever = MockAPIRetriever(chunks=mock_api_chunks)
    mock_generator = EvidenceGroundedGenerator(
        provider=MockLLMProvider(canned_response=canned_llm_response)
    )
    mock_pipeline = RAGPipeline(retriever=mock_retriever, generator=mock_generator)

    app.dependency_overrides[get_pipeline] = lambda: mock_pipeline
    client = TestClient(app)
    yield client
    app.dependency_overrides.clear()


def test_health_check_endpoint(api_client: TestClient):
    """Verify GET /health returns 200 and expected healthy status."""
    response = api_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "evidence-backed-rag"


def test_docs_and_openapi_endpoints(api_client: TestClient):
    """Verify OpenAPI documentation endpoints are exposed."""
    docs_resp = api_client.get("/docs")
    assert docs_resp.status_code == 200

    openapi_resp = api_client.get("/openapi.json")
    assert openapi_resp.status_code == 200
    assert "openapi" in openapi_resp.json()


def test_valid_query_endpoint(api_client: TestClient):
    """Verify POST /query returns 200 with grounded RAGResponse."""
    payload = {"question": "What is FastAPI?", "retriever_mode": "hybrid_rerank"}
    response = api_client.post("/query", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["sufficient_evidence"] is True
    assert "FastAPI is an asynchronous Python framework." in data["answer"]
    assert len(data["citations"]) == 1
    assert data["citations"][0]["chunk_id"] == "doc1#stratA#c0001"
    assert data["citations"][0]["text_snippet"] == "FastAPI is an asynchronous Python framework"
    assert data["confidence"] == 0.98
    assert "doc1#stratA#c0001" in data["retrieved_chunk_ids"]
    assert data["refusal_reason"] is None


def test_query_validation_empty_string(api_client: TestClient):
    """Verify POST /query rejects empty string question with 422 error."""
    payload = {"question": ""}
    response = api_client.post("/query", json=payload)
    assert response.status_code == 422


def test_query_validation_whitespace_only(api_client: TestClient):
    """Verify POST /query rejects whitespace-only question with 422 error."""
    payload = {"question": "   \n\t  "}
    response = api_client.post("/query", json=payload)
    assert response.status_code == 422


def test_query_validation_missing_question_field(api_client: TestClient):
    """Verify POST /query rejects missing question field with 422 error."""
    payload = {"retriever_mode": "dense_only"}
    response = api_client.post("/query", json=payload)
    assert response.status_code == 422


def test_query_refusal_on_unanswerable(api_client: TestClient):
    """Verify POST /query returns standard refusal when evidence is lacking."""
    payload = {"question": "What is unanswerable in this document?"}
    response = api_client.post("/query", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["sufficient_evidence"] is False
    assert data["answer"] == INSUFFICIENT_EVIDENCE_REFUSAL
    assert data["confidence"] == 0.0
    assert data["citations"] == []
    assert data["refusal_reason"] is not None


def test_query_preserves_citations_and_provenance(api_client: TestClient):
    """Verify citations preserve chunk IDs, text snippets, and provenance metadata."""
    payload = {"question": "Describe the stack.", "top_k": 5}
    response = api_client.post("/query", json=payload)
    assert response.status_code == 200

    data = response.json()
    citation = data["citations"][0]
    assert "chunk_id" in citation
    assert "text_snippet" in citation
    assert citation["chunk_id"] == "doc1#stratA#c0001"


def test_query_pipeline_failure_handling():
    """Verify unhandled pipeline errors return 500 without leaking stack traces."""
    class BrokenPipeline:
        def query(self, *args, **kwargs) -> RAGResponse:
            raise RuntimeError("Database connection suddenly dropped")

    app.dependency_overrides[get_pipeline] = lambda: BrokenPipeline()
    client = TestClient(app, raise_server_exceptions=False)

    try:
        response = client.post("/query", json={"question": "Trigger error"})
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
        # Ensure raw stack trace is not exposed
        assert "RuntimeError" not in data["detail"]
    finally:
        app.dependency_overrides.clear()
