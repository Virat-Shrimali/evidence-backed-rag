"""Document parsing, metadata extraction, and text normalization."""

import hashlib
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pypdf import PdfReader


@dataclass
class Page:
    """Represents an individual page within a parsed document."""
    page_number: int  # 1-indexed
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Document:
    """Parsed document entity with page breakdown and provenance metadata."""
    id: str
    content: str
    source: str
    file_path: str
    total_pages: int
    pages: list[Page]
    metadata: dict[str, Any] = field(default_factory=dict)


def clean_text(raw_text: str) -> str:
    """Normalize and clean raw extracted text.

    - Applies NFKC Unicode normalization.
    - Replaces typographical quotes and dashes with ASCII equivalents.
    - Removes non-printable control characters while preserving newlines and tabs.
    - Normalizes excessive blank lines and horizontal whitespace.
    """
    if not raw_text:
        return ""

    # Unicode NFKC normalization
    text = unicodedata.normalize("NFKC", raw_text)

    # Replace common typographic characters
    char_replacements = {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2014": "-",
        "\u2013": "-",
        "\u2026": "...",
        "\u00a0": " ",
        "\u2022": "*",
        "\r\n": "\n",
        "\r": "\n",
    }
    for orig, target in char_replacements.items():
        text = text.replace(orig, target)

    # Strip non-printable control characters (keep \n and \t)
    text = "".join(
        ch for ch in text
        if unicodedata.category(ch)[0] != "C" or ch in ("\n", "\t")
    )

    # Replace runs of horizontal whitespace with a single space
    text = re.sub(r"[ \t]+", " ", text)

    # Limit consecutive newlines to at most 2 (paragraph break)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)

    return text.strip()


def compute_file_sha256(file_path: Path) -> str:
    """Compute SHA-256 hash of a file for deterministic verification."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def generate_document_id(file_path: Path, content: str) -> str:
    """Generate a deterministic, filesystem-safe document identifier."""
    clean_stem = re.sub(r"[^\w\-]", "_", file_path.stem).lower()
    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()[:8]
    return f"doc_{clean_stem}_{content_hash}"


def parse_pdf(file_path: Path) -> Document:
    """Extract clean text and per-page metadata from a PDF file."""
    reader = PdfReader(str(file_path))
    total_pages = len(reader.pages)

    pages: list[Page] = []
    cleaned_page_texts: list[str] = []

    for idx, pdf_page in enumerate(reader.pages):
        page_num = idx + 1
        raw_page_text = pdf_page.extract_text() or ""
        cleaned = clean_text(raw_page_text)

        pages.append(
            Page(
                page_number=page_num,
                content=cleaned,
                metadata={"raw_char_count": len(raw_page_text)},
            )
        )
        if cleaned:
            cleaned_page_texts.append(cleaned)

    full_content = "\n\n".join(cleaned_page_texts)
    doc_id = generate_document_id(file_path, full_content)

    pdf_metadata = {}
    if reader.metadata:
        pdf_metadata = {
            "title": str(reader.metadata.title) if reader.metadata.title else None,
            "author": str(reader.metadata.author) if reader.metadata.author else None,
            "creator": str(reader.metadata.creator) if reader.metadata.creator else None,
        }

    return Document(
        id=doc_id,
        content=full_content,
        source=file_path.name,
        file_path=str(file_path.resolve()),
        total_pages=total_pages,
        pages=pages,
        metadata={
            "file_format": "pdf",
            "file_size_bytes": file_path.stat().st_size,
            "sha256": compute_file_sha256(file_path),
            **pdf_metadata,
        },
    )


def parse_text_file(file_path: Path) -> Document:
    """Extract and clean text from plain text or markdown files."""
    try:
        raw_text = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raw_text = file_path.read_text(encoding="latin-1", errors="replace")

    cleaned = clean_text(raw_text)
    doc_id = generate_document_id(file_path, cleaned)

    page = Page(page_number=1, content=cleaned, metadata={})
    return Document(
        id=doc_id,
        content=cleaned,
        source=file_path.name,
        file_path=str(file_path.resolve()),
        total_pages=1,
        pages=[page],
        metadata={
            "file_format": file_path.suffix.lstrip(".").lower() or "text",
            "file_size_bytes": file_path.stat().st_size,
            "sha256": compute_file_sha256(file_path),
        },
    )


def parse_document(file_path: Path) -> Document:
    """Dispatch file parsing based on extension."""
    if not file_path.is_file():
        raise FileNotFoundError(f"Document not found: {file_path}")

    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return parse_pdf(file_path)
    elif suffix in (".txt", ".md", ".csv", ".json", ".log"):
        return parse_text_file(file_path)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")


def parse_directory(directory_path: Path) -> list[Document]:
    """Recursively parse all supported documents in a directory."""
    documents: list[Document] = []
    if not directory_path.exists():
        return documents

    supported_extensions = {".pdf", ".txt", ".md"}
    for file_path in sorted(directory_path.glob("**/*")):
        if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
            documents.append(parse_document(file_path))

    return documents
