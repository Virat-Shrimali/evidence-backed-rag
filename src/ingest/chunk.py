"""Reproducible, configurable chunking strategies: Strategy A and Strategy B.

Algorithms & Reproducibility Specifications:
---------------------------------------------
Strategy A (Baseline Fixed-Size Token Chunking):
- Tokenizer: tiktoken cl100k_base
- Chunk size: 500 tokens
- Chunk overlap: 50 tokens (stride: 450 tokens)
- Algorithm: Sliding window across the token sequence of the document. Each window
  slice [i : i + 500] is decoded back to text. Token spans are mapped back to document
  page numbers.
- Deterministic Chunk ID: `{document_id}#stratA#c{chunk_index:04d}`

Strategy B (Fine-Grained Sentence-Preserving Chunking):
- Tokenizer: tiktoken cl100k_base
- Sentence Segmenter: pysbd (Python Sentence Boundary Disambiguation, English)
- Target chunk size: 200 tokens
- Target overlap: 20 tokens
- Algorithm: Document text is segmented into discrete grammatical sentences.
  Sentences are sequentially accumulated into a chunk buffer until adding the next
  sentence would exceed the 200-token ceiling.
  - Sentence preservation: Sentences are never cut mid-sentence, ensuring complete
    grammatical and semantic units.
  - Runaway sentence handling: If a single sentence exceeds 200 tokens (e.g., run-on
    legal clauses or unpunctuated lists), it is partitioned using a token sliding window
    so that no chunk exceeds 200 tokens.
  - Overlap: The trailing sentence(s) totaling up to target overlap are carried over
    to seed the next chunk buffer.
- Deterministic Chunk ID: `{document_id}#stratB#c{chunk_index:04d}`
"""

from dataclasses import dataclass, field
from typing import Any

import pysbd

from src.config import ChunkingConfig, settings
from src.ingest.parse import Document, Page
from src.ingest.tokenizer import count_tokens, decode_tokens, encode_text


@dataclass
class Chunk:
    """A discrete verifiable text chunk with provenance and token metadata."""
    id: str
    document_id: str
    content: str
    token_count: int
    chunk_index: int
    page_numbers: list[int] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def _find_page_numbers_for_char_range(
    pages: list[Page],
    start_char: int,
    end_char: int,
) -> list[int]:
    """Determine which 1-indexed document page numbers span the given character offsets."""
    if not pages:
        return [1]

    overlapping_pages: list[int] = []
    current_offset = 0

    for page in pages:
        page_len = len(page.content)
        page_start = current_offset
        page_end = current_offset + page_len

        # Check if the page text overlaps with the char range
        if max(start_char, page_start) < min(end_char, page_end):
            overlapping_pages.append(page.page_number)

        # +2 accounts for the '\n\n' delimiter joining pages in Document.content
        current_offset = page_end + 2

    return overlapping_pages or [pages[0].page_number]


def chunk_strategy_a(
    document: Document,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    encoding_name: str | None = None,
) -> list[Chunk]:
    """Strategy A: Fixed-size token chunking with sliding window."""
    if not document.content.strip():
        return []

    tokens = encode_text(document.content, encoding_name=encoding_name)
    if not tokens:
        return []

    stride = max(1, chunk_size - chunk_overlap)
    chunks: list[Chunk] = []
    chunk_idx = 0

    # Locate character boundaries for page number mapping
    full_text = document.content

    for start_tok in range(0, len(tokens), stride):
        end_tok = min(start_tok + chunk_size, len(tokens))
        chunk_token_slice = tokens[start_tok:end_tok]
        chunk_text = decode_tokens(chunk_token_slice, encoding_name=encoding_name).strip()

        if not chunk_text:
            continue

        # Find character offset in document to resolve page numbers
        char_start = full_text.find(chunk_text[:50]) if len(chunk_text) >= 50 else full_text.find(chunk_text)
        char_start = max(0, char_start)
        char_end = char_start + len(chunk_text)
        page_nums = _find_page_numbers_for_char_range(document.pages, char_start, char_end)

        chunk_id = f"{document.id}#stratA#c{chunk_idx:04d}"
        chunks.append(
            Chunk(
                id=chunk_id,
                document_id=document.id,
                content=chunk_text,
                token_count=len(chunk_token_slice),
                chunk_index=chunk_idx,
                page_numbers=page_nums,
                metadata={
                    "strategy": "strategy_a",
                    "chunk_size_tokens": chunk_size,
                    "chunk_overlap_tokens": chunk_overlap,
                    "start_token": start_tok,
                    "end_token": end_tok,
                    "source": document.source,
                },
            )
        )
        chunk_idx += 1

        if end_tok >= len(tokens):
            break

    return chunks


def _split_long_sentence(
    sentence: str,
    max_tokens: int,
    overlap_tokens: int,
    encoding_name: str | None,
) -> list[str]:
    """Partition a runaway sentence that exceeds max_tokens into smaller token windows."""
    tokens = encode_text(sentence, encoding_name=encoding_name)
    if len(tokens) <= max_tokens:
        return [sentence]

    stride = max(1, max_tokens - overlap_tokens)
    sub_sentences = []
    for i in range(0, len(tokens), stride):
        sub_tokens = tokens[i : i + max_tokens]
        decoded = decode_tokens(sub_tokens, encoding_name=encoding_name).strip()
        if decoded:
            sub_sentences.append(decoded)
        if i + max_tokens >= len(tokens):
            break
    return sub_sentences


def chunk_strategy_b(
    document: Document,
    target_chunk_size: int = 200,
    target_overlap: int = 20,
    encoding_name: str | None = None,
) -> list[Chunk]:
    """Strategy B: Fine-grained 200-token chunking preserving sentence boundaries."""
    if not document.content.strip():
        return []

    segmenter = pysbd.Segmenter(language="en", clean=False)
    raw_sentences = segmenter.segment(document.content)

    # Flatten and split any single sentences that exceed target_chunk_size
    sentences: list[str] = []
    for s in raw_sentences:
        s_clean = s.strip()
        if not s_clean:
            continue
        s_tok_count = count_tokens(s_clean, encoding_name=encoding_name)
        if s_tok_count > target_chunk_size:
            sentences.extend(
                _split_long_sentence(
                    s_clean,
                    max_tokens=target_chunk_size,
                    overlap_tokens=target_overlap,
                    encoding_name=encoding_name,
                )
            )
        else:
            sentences.append(s_clean)

    if not sentences:
        return []

    chunks: list[Chunk] = []
    full_text = document.content
    chunk_idx = 0
    i = 0

    while i < len(sentences):
        current_sentences: list[str] = []
        current_tokens = 0

        # Accumulate whole sentences up to target_chunk_size
        j = i
        while j < len(sentences):
            next_sent = sentences[j]
            next_sent_tok = count_tokens(next_sent, encoding_name=encoding_name)

            if current_sentences and (current_tokens + next_sent_tok > target_chunk_size):
                break

            current_sentences.append(next_sent)
            current_tokens += next_sent_tok
            j += 1

        if not current_sentences:
            # Single sentence was somehow skipped or handled
            current_sentences.append(sentences[i])
            j = i + 1

        chunk_text = " ".join(current_sentences).strip()
        actual_tok_count = count_tokens(chunk_text, encoding_name=encoding_name)

        # Map to page numbers
        char_start = full_text.find(chunk_text[:50]) if len(chunk_text) >= 50 else full_text.find(chunk_text)
        char_start = max(0, char_start)
        char_end = char_start + len(chunk_text)
        page_nums = _find_page_numbers_for_char_range(document.pages, char_start, char_end)

        chunk_id = f"{document.id}#stratB#c{chunk_idx:04d}"
        chunks.append(
            Chunk(
                id=chunk_id,
                document_id=document.id,
                content=chunk_text,
                token_count=actual_tok_count,
                chunk_index=chunk_idx,
                page_numbers=page_nums,
                metadata={
                    "strategy": "strategy_b",
                    "target_chunk_size": target_chunk_size,
                    "target_overlap": target_overlap,
                    "sentence_count": len(current_sentences),
                    "source": document.source,
                },
            )
        )
        chunk_idx += 1

        if j >= len(sentences):
            break

        # Calculate overlap for next iteration: keep trailing sentence(s) within target_overlap
        overlap_tokens_accum = 0
        overlap_count = 0
        for sent in reversed(current_sentences):
            sent_tok = count_tokens(sent, encoding_name=encoding_name)
            if overlap_tokens_accum + sent_tok <= target_overlap or overlap_count == 0:
                overlap_tokens_accum += sent_tok
                overlap_count += 1
            else:
                break

        # Avoid infinite loop: advance by at least 1 new sentence
        advance = max(1, len(current_sentences) - overlap_count)
        i += advance

    return chunks


def chunk_document(
    document: Document,
    strategy: str = "strategy_a",
    config: ChunkingConfig | None = None,
) -> list[Chunk]:
    """Chunk a document according to the specified strategy and configuration."""
    cfg = config or settings.chunking
    if strategy == "strategy_b":
        return chunk_strategy_b(
            document=document,
            target_chunk_size=cfg.strategy_b_chunk_size,
            target_overlap=cfg.strategy_b_chunk_overlap,
            encoding_name=cfg.tokenizer_encoding,
        )
    else:
        return chunk_strategy_a(
            document=document,
            chunk_size=cfg.strategy_a_chunk_size,
            chunk_overlap=cfg.strategy_a_chunk_overlap,
            encoding_name=cfg.tokenizer_encoding,
        )
