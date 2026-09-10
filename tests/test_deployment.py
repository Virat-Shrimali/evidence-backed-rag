"""Tests for deployment readiness, container configuration, and environment variable bindings."""

from pathlib import Path

from fastapi.testclient import TestClient

from api.main import app
from src.config import RAGConfig


def test_docker_and_deployment_files_exist():
    """Verify that Dockerfile and .dockerignore exist and contain essential configuration."""
    base_dir = Path(__file__).resolve().parent.parent
    dockerfile = base_dir / "Dockerfile"
    dockerignore = base_dir / ".dockerignore"

    assert dockerfile.exists(), "Dockerfile must exist in repository root."
    assert dockerignore.exists(), ".dockerignore must exist in repository root."

    df_content = dockerfile.read_text(encoding="utf-8")
    assert "python:3.11-slim" in df_content, "Dockerfile must use Python 3.11-slim."
    assert "useradd" in df_content and "1000" in df_content, "Dockerfile must configure UID 1000 user for HF Spaces."
    assert "USER user" in df_content, "Dockerfile must run as non-root user."
    assert "EXPOSE 7860" in df_content, "Dockerfile must expose port 7860 for HF Spaces."
    assert "uvicorn" in df_content, "Dockerfile must run uvicorn api.main:app."

    di_content = dockerignore.read_text(encoding="utf-8")
    assert ".git" in di_content
    assert ".venv" in di_content
    assert "chroma_db" in di_content
    assert ".env" in di_content


def test_rag_config_port_and_env_binding(monkeypatch):
    """Verify that PORT and LLM_PROVIDER environment variables configure RAGConfig appropriately."""
    monkeypatch.setenv("PORT", "7860")
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.setenv("LLM_MODEL_NAME", "test-model")

    config = RAGConfig()
    assert config.api_port == 7860
    assert config.llm_provider == "mock"
    assert config.llm_model_name == "test-model"


def test_health_check_runs_without_external_dependencies():
    """Verify GET /health responds with 200 without requiring LLM provider or vector stores."""
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "evidence-backed-rag"
