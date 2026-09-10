"""Unit tests for document parsing, text normalization, and metadata extraction."""

from pathlib import Path

import pytest
from pypdf import PdfWriter

from src.ingest.parse import (
    clean_text,
    compute_file_sha256,
    generate_document_id,
    parse_directory,
    parse_document,
    parse_pdf,
    parse_text_file,
)


@pytest.fixture
def temp_text_file(tmp_path: Path) -> Path:
    """Create a temporary text file with sample content."""
    file_path = tmp_path / "sample_policy.txt"
    file_path.write_text(
        "Quarterly Financial Statement\n\n"
        "Revenue increased by 15% across all regions. Operating margins improved to 22%.\n\n"
        "Primary risks include currency fluctuations and supply chain delays.",
        encoding="utf-8",
    )
    return file_path


@pytest.fixture
def temp_pdf_file(tmp_path: Path) -> Path:
    """Generate a valid multi-page PDF file in memory for ingestion tests."""
    writer = PdfWriter()
    # Add page 1
    writer.add_blank_page(width=200, height=200)
    # Add page 2
    writer.add_blank_page(width=200, height=200)

    pdf_path = tmp_path / "sample_multi_page.pdf"
    with open(pdf_path, "wb") as f:
        writer.write(f)

    return pdf_path


def test_clean_text_normalizes_quotes_dashes_and_spaces():
    """Verify smart quotes, em-dashes, and irregular whitespace are cleaned."""
    raw = "“Smart quotes” and ‘single quotes’ — along with\u00a0non-breaking spaces   and\r\ntabs."
    cleaned = clean_text(raw)
    assert cleaned == '"Smart quotes" and \'single quotes\' - along with non-breaking spaces and\ntabs.'


def test_clean_text_limits_consecutive_blank_lines():
    """Verify clean_text collapses excessive blank lines to at most two."""
    raw = "Paragraph 1\n\n\n\n\nParagraph 2\n\n\nParagraph 3"
    cleaned = clean_text(raw)
    assert cleaned == "Paragraph 1\n\nParagraph 2\n\nParagraph 3"


def test_clean_text_empty_and_whitespace():
    """Verify clean_text handles empty or whitespace-only strings."""
    assert clean_text("") == ""
    assert clean_text("   \n\t  \n  ") == ""


def test_compute_file_sha256(temp_text_file: Path):
    """Verify SHA-256 hash calculation is deterministic and non-empty."""
    hash1 = compute_file_sha256(temp_text_file)
    hash2 = compute_file_sha256(temp_text_file)
    assert hash1 == hash2
    assert len(hash1) == 64


def test_generate_document_id():
    """Verify generated document ID is deterministic, lowercase, and contains hash."""
    path = Path("reports/Annual Report 2023.PDF")
    content = "Sample report text"
    doc_id1 = generate_document_id(path, content)
    doc_id2 = generate_document_id(path, content)
    assert doc_id1 == doc_id2
    assert doc_id1.startswith("doc_annual_report_2023_")


def test_parse_text_file(temp_text_file: Path):
    """Verify text file parsing extracts content, single page, and metadata."""
    doc = parse_text_file(temp_text_file)
    assert doc.source == "sample_policy.txt"
    assert doc.total_pages == 1
    assert len(doc.pages) == 1
    assert doc.pages[0].page_number == 1
    assert "Revenue increased by 15%" in doc.content
    assert doc.metadata["file_format"] == "txt"
    assert doc.metadata["file_size_bytes"] > 0
    assert len(doc.metadata["sha256"]) == 64


def test_parse_pdf_metadata_and_pages(temp_pdf_file: Path):
    """Verify PDF parser captures total pages and page numbers."""
    doc = parse_pdf(temp_pdf_file)
    assert doc.source == "sample_multi_page.pdf"
    assert doc.total_pages == 2
    assert len(doc.pages) == 2
    assert doc.pages[0].page_number == 1
    assert doc.pages[1].page_number == 2
    assert doc.metadata["file_format"] == "pdf"


def test_parse_document_dispatcher(temp_text_file: Path, temp_pdf_file: Path):
    """Verify parse_document dispatches correctly based on file extension."""
    doc_txt = parse_document(temp_text_file)
    assert doc_txt.metadata["file_format"] == "txt"

    doc_pdf = parse_document(temp_pdf_file)
    assert doc_pdf.metadata["file_format"] == "pdf"


def test_parse_document_invalid_file(tmp_path: Path):
    """Verify parse_document raises appropriate errors for missing or unsupported files."""
    with pytest.raises(FileNotFoundError):
        parse_document(tmp_path / "non_existent_file.txt")

    unsupported = tmp_path / "test.exe"
    unsupported.write_bytes(b"binary data")
    with pytest.raises(ValueError, match="Unsupported file format"):
        parse_document(unsupported)


def test_parse_directory(tmp_path: Path, temp_text_file: Path):
    """Verify directory parsing scans supported documents recursively."""
    sub_dir = tmp_path / "nested"
    sub_dir.mkdir()
    (sub_dir / "nested_doc.md").write_text("# Markdown Title\nSome content.", encoding="utf-8")

    docs = parse_directory(tmp_path)
    assert len(docs) >= 2
    doc_names = [d.source for d in docs]
    assert "sample_policy.txt" in doc_names
    assert "nested_doc.md" in doc_names
