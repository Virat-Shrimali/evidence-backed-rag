"""Sparse indexing using Okapi BM25."""


from src.ingest.chunk import Chunk


class BM25Index:
    """Sparse index wrapper around rank_bm25 or lightweight tokenized lookup."""

    def __init__(self, chunks: list[Chunk] | None = None):
        self.chunks: dict[str, Chunk] = {}
        self.corpus: list[list[str]] = []
        self.chunk_ids: list[str] = []
        self._bm25 = None
        if chunks:
            self.index_chunks(chunks)

    def _tokenize(self, text: str) -> list[str]:
        """Simple whitespace and lowercasing tokenizer."""
        return [word.strip(".,!?;:()[]{}\"'").lower() for word in text.split() if word.strip()]

    def index_chunks(self, chunks: list[Chunk]) -> None:
        """Build BM25 index over provided chunks."""
        self.chunks = {c.id: c for c in chunks}
        self.chunk_ids = [c.id for c in chunks]
        self.corpus = [self._tokenize(c.content) for c in chunks]
        try:
            from rank_bm25 import BM25Okapi
            self._bm25 = BM25Okapi(self.corpus)
        except ImportError:
            # Fallback simple keyword frequency scorer if rank_bm25 not yet installed
            self._bm25 = None

    def search(self, query: str, top_k: int = 5) -> list[str]:
        """Return list of top_k chunk_ids matching query."""
        if not self.chunk_ids:
            return []

        tokenized_query = self._tokenize(query)
        if self._bm25 is not None:
            doc_scores = self._bm25.get_scores(tokenized_query)
            scored = list(zip(self.chunk_ids, doc_scores, strict=False))
            scored.sort(key=lambda x: x[1], reverse=True)
            return [chunk_id for chunk_id, score in scored[:top_k] if score > 0]

        # Lightweight fallback token overlap scoring
        q_tokens = set(tokenized_query)
        scored = []
        for chunk_id, tokens in zip(self.chunk_ids, self.corpus, strict=False):
            score = len(q_tokens.intersection(tokens))
            scored.append((chunk_id, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [chunk_id for chunk_id, score in scored[:top_k] if score > 0]
