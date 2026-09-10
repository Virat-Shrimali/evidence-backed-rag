"""Document parsing and text normalization module."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Document:
    """Parsed document entity with provenance metadata."""
    id: str
    content: str
    source: str
    metadata: dict


def parse_text_file(file_path: Path) -> Document:
    """Parse a plain text file into a Document."""
    content = file_path.read_text(encoding="utf-8")
    return Document(
        id=file_path.stem,
        content=content.strip(),
        source=file_path.name,
        metadata={"file_path": str(file_path)},
    )


def parse_directory(directory_path: Path) -> list[Document]:
    """Parse all supported text documents from a directory."""
    documents: list[Document] = []
    if not directory_path.exists():
        return documents

    for file_path in directory_path.glob("**/*"):
        if file_path.is_file() and file_path.suffix.lower() in [".txt", ".md"]:
            documents.append(parse_text_file(file_path))
    return documents
