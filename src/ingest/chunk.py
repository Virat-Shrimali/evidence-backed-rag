"""Chunking module supporting Strategy A (fixed 500) and Strategy B (200 tokens)."""

from dataclasses import dataclass

from src.config import settings
from src.ingest.parse import Document


@dataclass
class Chunk:
    """A chunk of text extracted from a Document with verifiable citation id."""
    id: str
    document_id: str
    content: str
    chunk_index: int
    metadata: dict


def chunk_text_sliding_window(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[str]:
    """Simple character/word sliding window chunker."""
    words = text.split()
    if not words:
        return []

    chunks = []
    step = max(1, chunk_size - chunk_overlap)
    for i in range(0, len(words), step):
        chunk_words = words[i : i + chunk_size]
        chunks.append(" ".join(chunk_words))
        if i + chunk_size >= len(words):
            break
    return chunks


def chunk_document(
    document: Document,
    strategy: str = "strategy_a",
) -> list[Chunk]:
    """Chunk a document according to the specified strategy."""
    if strategy == "strategy_b":
        chunk_size = settings.chunking.strategy_b_chunk_size
        chunk_overlap = settings.chunking.strategy_b_chunk_overlap
    else:
        chunk_size = settings.chunking.strategy_a_chunk_size
        chunk_overlap = settings.chunking.strategy_a_chunk_overlap

    raw_chunks = chunk_text_sliding_window(
        document.content,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks: list[Chunk] = []
    for idx, content in enumerate(raw_chunks):
        chunk_id = f"{document.id}#chunk-{idx}"
        chunk = Chunk(
            id=chunk_id,
            document_id=document.id,
            content=content,
            chunk_index=idx,
            metadata={
                **document.metadata,
                "strategy": strategy,
                "word_count": len(content.split()),
            },
        )
        chunks.append(chunk)

    return chunks
