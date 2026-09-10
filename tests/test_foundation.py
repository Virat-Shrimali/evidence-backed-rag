"""Foundation and environment verification tests."""

from fastapi.testclient import TestClient

from api.main import app
from src.config import INSUFFICIENT_EVIDENCE_REFUSAL, settings


def test_configuration_loading():
    """Verify default configuration attributes load with valid types."""
    assert settings.embedding_model_name == "sentence-transformers/all-MiniLM-L6-v2"
    assert settings.chunking.strategy_a_chunk_size == 500
    assert settings.chunking.strategy_b_chunk_size == 200
    assert settings.refusal_message == INSUFFICIENT_EVIDENCE_REFUSAL
    assert settings.evidence_confidence_threshold > 0.0


def test_api_health_endpoint():
    """Verify FastAPI health check returns 200 and healthy status."""
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "service": "evidence-backed-rag"}
