"""Pytest fixtures and test environment setup."""

import pytest

from src.ingest.chunk import Chunk
from src.ingest.parse import Document


@pytest.fixture
def sample_document():
    """Provides a sample parsed document for testing."""
    return Document(
        id="sample_doc",
        content=(
            "The Company recorded annual revenue of $120 million in fiscal year 2023. "
            "Operating margins expanded by 3.5% driven by automation efficiencies. "
            "Foreign exchange volatility presents the primary risk factor for the upcoming quarter."
        ),
        source="sample.txt",
        metadata={"category": "financial_report"},
    )


@pytest.fixture
def sample_chunks():
    """Provides sample chunks for indexing and retrieval tests."""
    return [
        Chunk(
            id="doc1#chunk-0",
            document_id="doc1",
            content="Annual revenue reached 120 million dollars with strong expansion.",
            chunk_index=0,
            metadata={"source": "doc1"},
        ),
        Chunk(
            id="doc1#chunk-1",
            document_id="doc1",
            content="Operating margin increased 3.5 percent due to automation.",
            chunk_index=1,
            metadata={"source": "doc1"},
        ),
        Chunk(
            id="doc2#chunk-0",
            document_id="doc2",
            content="Risk factors include currency exchange fluctuations and supplier delays.",
            chunk_index=0,
            metadata={"source": "doc2"},
        ),
    ]
