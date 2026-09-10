"""Unit tests for document chunking strategies."""

from src.ingest.chunk import chunk_document, chunk_text_sliding_window
from src.ingest.parse import Document


def test_chunk_text_sliding_window():
    """Verify sliding window chunker splits text correctly with overlap."""
    text = "one two three four five six seven eight nine ten"
    chunks = chunk_text_sliding_window(text, chunk_size=5, chunk_overlap=2)
    assert len(chunks) >= 2
    assert chunks[0] == "one two three four five"
    # Step = 5 - 2 = 3. Next chunk starts at index 3: four five six seven eight
    assert chunks[1] == "four five six seven eight"


def test_chunk_document_strategy_a(sample_document: Document):
    """Verify Strategy A produces chunks with valid IDs and metadata."""
    chunks = chunk_document(sample_document, strategy="strategy_a")
    assert len(chunks) > 0
    assert chunks[0].id.startswith(f"{sample_document.id}#chunk-")
    assert chunks[0].metadata["strategy"] == "strategy_a"
    assert chunks[0].document_id == sample_document.id


def test_chunk_document_strategy_b(sample_document: Document):
    """Verify Strategy B produces chunks with smaller sizes."""
    chunks = chunk_document(sample_document, strategy="strategy_b")
    assert len(chunks) > 0
    assert chunks[0].metadata["strategy"] == "strategy_b"
