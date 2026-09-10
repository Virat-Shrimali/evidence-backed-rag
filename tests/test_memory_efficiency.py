"""Tests for memory efficiency, lazy model loading, and resource reuse."""

from unittest.mock import MagicMock

import pytest

from src.config import RAGConfig
from src.pipeline import RAGPipeline
from src.retrieval.retriever import (
    BM25Retriever,
    clear_retriever_cache,
    create_retriever,
)


@pytest.fixture(autouse=True)
def reset_cache():
    """Ensure clean cache before and after every test."""
    clear_retriever_cache()
    yield
    clear_retriever_cache()


def test_bm25_only_does_not_instantiate_dense_embedding_model(monkeypatch):
    """Verify bm25_only strategy does NOT initialize DenseIndex or SentenceTransformers."""

    def forbidden_dense_init(*args, **kwargs):
        pytest.fail("DenseIndex was initialized during bm25_only strategy!")

    monkeypatch.setattr("src.index.embed.DenseIndex.__init__", forbidden_dense_init)

    retriever = create_retriever(strategy="bm25_only")
    assert isinstance(retriever, BM25Retriever)
    results = retriever.retrieve("Evidence-Backed RAG", top_k=3)
    assert len(results) > 0
    assert all(r.retrieval_method == "bm25" for r in results)


def test_bm25_only_does_not_instantiate_cross_encoder(monkeypatch):
    """Verify bm25_only strategy does NOT initialize CrossEncoderReranker."""

    def forbidden_reranker_init(*args, **kwargs):
        pytest.fail("CrossEncoderReranker was initialized during bm25_only strategy!")

    monkeypatch.setattr(
        "src.retrieval.rerank.CrossEncoderReranker.__init__",
        forbidden_reranker_init,
    )

    retriever = create_retriever(strategy="bm25_only")
    assert isinstance(retriever, BM25Retriever)
    results = retriever.retrieve("Evidence-Backed RAG", top_k=3)
    assert len(results) > 0


def test_dense_only_does_not_instantiate_cross_encoder(monkeypatch):
    """Verify dense_only strategy does NOT initialize CrossEncoderReranker."""

    def forbidden_reranker_init(*args, **kwargs):
        pytest.fail("CrossEncoderReranker was initialized during dense_only strategy!")

    monkeypatch.setattr(
        "src.retrieval.rerank.CrossEncoderReranker.__init__",
        forbidden_reranker_init,
    )

    mock_dense = MagicMock()
    mock_dense.search.return_value = []
    retriever = create_retriever(strategy="dense_only", dense_index=mock_dense)
    retriever.retrieve("test query")
    mock_dense.search.assert_called_once()


def test_retriever_resource_caching_and_reuse():
    """Verify that subsequent calls to create_retriever reuse the shared index rather than re-parsing/re-indexing."""
    retriever1 = create_retriever(strategy="bm25_only")
    retriever2 = create_retriever(strategy="bm25_only")

    # Underlying BM25Index instance must be identical
    assert retriever1.bm25_index is retriever2.bm25_index
    assert retriever1.bm25_index.count() > 0


def test_pipeline_caches_retriever_instances():
    """Verify RAGPipeline caches retriever per strategy mode."""
    pipeline = RAGPipeline()
    ret1 = pipeline.get_retriever("bm25_only")
    ret2 = pipeline.get_retriever("bm25_only")

    assert ret1 is ret2


def test_render_environment_defaults_to_bm25_only(monkeypatch):
    """Verify that in a Render environment (RENDER=true), default retrieval strategy is bm25_only."""
    monkeypatch.setenv("RENDER", "true")
    monkeypatch.delenv("RETRIEVAL_STRATEGY", raising=False)

    config = RAGConfig()
    assert config.retrieval_strategy == "bm25_only"


def test_render_environment_respects_explicit_override(monkeypatch):
    """Verify that setting RETRIEVAL_STRATEGY explicitly overrides the Render default."""
    monkeypatch.setenv("RENDER", "true")
    monkeypatch.setenv("RETRIEVAL_STRATEGY", "hybrid")

    config = RAGConfig()
    assert config.retrieval_strategy == "hybrid"


def test_low_memory_mode_defaults_to_bm25_only(monkeypatch):
    """Verify LOW_MEMORY_MODE=true sets default retrieval strategy to bm25_only."""
    monkeypatch.delenv("RENDER", raising=False)
    monkeypatch.setenv("LOW_MEMORY_MODE", "true")
    monkeypatch.delenv("RETRIEVAL_STRATEGY", raising=False)

    config = RAGConfig()
    assert config.retrieval_strategy == "bm25_only"
