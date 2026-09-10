"""Deterministic tokenization utility using tiktoken."""

from functools import lru_cache

import tiktoken

from src.config import settings


@lru_cache(maxsize=4)
def get_tokenizer(encoding_name: str | None = None) -> tiktoken.Encoding:
    """Get cached tiktoken encoding instance."""
    name = encoding_name or settings.chunking.tokenizer_encoding
    return tiktoken.get_encoding(name)


def count_tokens(text: str, encoding_name: str | None = None) -> int:
    """Count tokens in text."""
    if not text:
        return 0
    enc = get_tokenizer(encoding_name)
    return len(enc.encode(text))


def encode_text(text: str, encoding_name: str | None = None) -> list[int]:
    """Encode text to token IDs."""
    if not text:
        return []
    enc = get_tokenizer(encoding_name)
    return enc.encode(text)


def decode_tokens(tokens: list[int], encoding_name: str | None = None) -> str:
    """Decode token IDs back to text."""
    if not tokens:
        return ""
    enc = get_tokenizer(encoding_name)
    return enc.decode(tokens)
