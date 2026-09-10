"""Pytest fixtures and test environment setup."""

import pytest

from src.ingest.chunk import Chunk
from src.ingest.parse import Document, Page


@pytest.fixture
def sample_document():
    """Provides a sample parsed document for testing."""
    content = (
        "The Company recorded annual revenue of $120 million in fiscal year 2023. "
        "Operating margins expanded by 3.5% driven by automation efficiencies. "
        "Foreign exchange volatility presents the primary risk factor for the upcoming quarter."
    )
    return Document(
        id="sample_doc",
        content=content,
        source="sample.txt",
        file_path="/data/sample.txt",
        total_pages=1,
        pages=[Page(page_number=1, content=content)],
        metadata={"category": "financial_report"},
    )


@pytest.fixture
def multi_page_document():
    """Provides a multi-page document for page number tracking tests."""
    p1 = "First page content. Dr. Smith reviewed the quarterly results on Jan. 1st."
    p2 = "Second page content. Operating expenses dropped by 12 percent overall."
    full = f"{p1}\n\n{p2}"
    return Document(
        id="multi_doc",
        content=full,
        source="multi.pdf",
        file_path="/data/multi.pdf",
        total_pages=2,
        pages=[
            Page(page_number=1, content=p1),
            Page(page_number=2, content=p2),
        ],
        metadata={"file_format": "pdf"},
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
