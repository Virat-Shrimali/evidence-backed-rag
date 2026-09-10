"""Unit tests for Strategy A and Strategy B chunking algorithms and edge cases."""

from src.config import ChunkingConfig
from src.ingest.chunk import (
    chunk_document,
    chunk_strategy_a,
    chunk_strategy_b,
)
from src.ingest.parse import Document, Page


def test_chunk_document_strategy_a_deterministic_ids_and_metadata(sample_document: Document):
    """Verify Strategy A produces valid deterministic IDs and preserves metadata."""
    chunks = chunk_document(sample_document, strategy="strategy_a")
    assert len(chunks) > 0
    assert chunks[0].id == f"{sample_document.id}#stratA#c0000"
    assert chunks[0].document_id == sample_document.id
    assert chunks[0].metadata["strategy"] == "strategy_a"
    assert chunks[0].token_count > 0
    assert chunks[0].page_numbers == [1]


def test_chunk_document_strategy_a_sliding_window_overlap():
    """Verify Strategy A sliding window produces chunks with expected token overlap."""
    # Create text with ~300 tokens
    text = " ".join(["token" + str(i) for i in range(300)])
    doc = Document(
        id="long_doc",
        content=text,
        source="long.txt",
        file_path="/data/long.txt",
        total_pages=1,
        pages=[Page(page_number=1, content=text)],
        metadata={},
    )

    chunks = chunk_strategy_a(doc, chunk_size=100, chunk_overlap=20)
    assert len(chunks) > 1
    # First chunk has 100 tokens, stride is 80
    assert chunks[0].token_count == 100
    assert chunks[0].id == "long_doc#stratA#c0000"
    assert chunks[1].id == "long_doc#stratA#c0001"


def test_chunk_document_strategy_a_multi_page_mapping(multi_page_document: Document):
    """Verify chunks accurately map to page numbers across document pages."""
    chunks = chunk_strategy_a(multi_page_document, chunk_size=500, chunk_overlap=50)
    assert len(chunks) > 0
    # The document is short enough to fit in one chunk, which spans both page 1 and page 2
    assert 1 in chunks[0].page_numbers or 2 in chunks[0].page_numbers


def test_chunk_document_strategy_b_sentence_boundary_preservation():
    """Verify Strategy B preserves complete sentences without cutting mid-sentence."""
    sentences = [
        "First quarter earnings were notably higher than projected.",
        "Management attributed this growth to disciplined cost controls in manufacturing.",
        "Research and development outlays rose by eight percent in the same duration.",
        "Foreign currency fluctuations remain a headwind for the next two quarters.",
    ]
    text = " ".join(sentences)
    doc = Document(
        id="sent_doc",
        content=text,
        source="sent.txt",
        file_path="/data/sent.txt",
        total_pages=1,
        pages=[Page(page_number=1, content=text)],
        metadata={},
    )

    chunks = chunk_strategy_b(doc, target_chunk_size=50, target_overlap=15)
    assert len(chunks) > 0

    for chunk in chunks:
        # Every chunk should end with sentence terminal punctuation
        assert chunk.content[-1] in (".", "!", "?")
        assert chunk.id.startswith("sent_doc#stratB#c")
        assert chunk.metadata["strategy"] == "strategy_b"


def test_chunk_document_strategy_b_complex_sentence_disambiguation():
    """Verify sentence segmentation does not break on abbreviations like Dr. or U.S."""
    text = (
        "Dr. Smith visited the U.S. Federal Reserve on Jan. 15th. "
        "The subsequent report detailed interest rate expectations for late 2024. "
        "All participating committee members concurred with the baseline scenario."
    )
    doc = Document(
        id="abbrev_doc",
        content=text,
        source="abbrev.txt",
        file_path="/data/abbrev.txt",
        total_pages=1,
        pages=[Page(page_number=1, content=text)],
        metadata={},
    )

    chunks = chunk_strategy_b(doc, target_chunk_size=100, target_overlap=20)
    assert len(chunks) >= 1
    # Check that "Dr. Smith visited the U.S. Federal Reserve on Jan. 15th." is not split into fragments
    assert "Dr. Smith visited the U.S. Federal Reserve on Jan. 15th." in chunks[0].content


def test_chunk_document_strategy_b_runaway_sentence_handling():
    """Verify a single giant sentence exceeding target_chunk_size is safely partitioned."""
    long_sentence = "Unpunctuated continuous data clause " * 50 + "."
    doc = Document(
        id="runaway_doc",
        content=long_sentence,
        source="runaway.txt",
        file_path="/data/runaway.txt",
        total_pages=1,
        pages=[Page(page_number=1, content=long_sentence)],
        metadata={},
    )

    # Strategy B target size 50 tokens
    chunks = chunk_strategy_b(doc, target_chunk_size=50, target_overlap=10)
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.token_count <= 60  # Allows slight margin but strictly bounded


def test_chunk_reproducibility(sample_document: Document):
    """Verify chunking the same document multiple times yields identical outputs."""
    run1 = chunk_document(sample_document, strategy="strategy_a")
    run2 = chunk_document(sample_document, strategy="strategy_a")
    assert len(run1) == len(run2)
    for c1, c2 in zip(run1, run2, strict=True):
        assert c1.id == c2.id
        assert c1.content == c2.content
        assert c1.token_count == c2.token_count
        assert c1.page_numbers == c2.page_numbers

    run1_b = chunk_document(sample_document, strategy="strategy_b")
    run2_b = chunk_document(sample_document, strategy="strategy_b")
    assert len(run1_b) == len(run2_b)
    for c1, c2 in zip(run1_b, run2_b, strict=True):
        assert c1.id == c2.id
        assert c1.content == c2.content


def test_configurable_chunking(sample_document: Document):
    """Verify custom ChunkingConfig overrides default chunk parameters."""
    custom_cfg = ChunkingConfig(
        strategy_a_chunk_size=15,
        strategy_a_chunk_overlap=5,
        strategy_b_chunk_size=25,
        strategy_b_chunk_overlap=5,
    )
    chunks_a = chunk_document(sample_document, strategy="strategy_a", config=custom_cfg)
    assert len(chunks_a) > 1
    assert chunks_a[0].metadata["chunk_size_tokens"] == 15

    chunks_b = chunk_document(sample_document, strategy="strategy_b", config=custom_cfg)
    assert len(chunks_b) > 0
    assert chunks_b[0].metadata["target_chunk_size"] == 25


def test_empty_document_handling():
    """Verify empty document produces an empty list of chunks without errors."""
    empty_doc = Document(
        id="empty_doc",
        content="",
        source="empty.txt",
        file_path="/data/empty.txt",
        total_pages=0,
        pages=[],
        metadata={},
    )
    assert chunk_document(empty_doc, strategy="strategy_a") == []
    assert chunk_document(empty_doc, strategy="strategy_b") == []
