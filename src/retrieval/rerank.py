"""Cross-encoder reranking module (Tier S)."""


from src.config import settings
from src.ingest.chunk import Chunk


class CrossEncoderReranker:
    """Reranker using cross-encoder/ms-marco-MiniLM-L-6-v2."""

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or settings.reranker_model_name
        self._model = None

    def _load_model(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self.model_name)
            except ImportError:
                self._model = None
        return self._model

    def rerank(
        self,
        query: str,
        chunks: list[Chunk],
        top_k: int = 5,
    ) -> list[tuple[Chunk, float]]:
        """Rerank candidates using CrossEncoder scores."""
        if not chunks:
            return []

        model = self._load_model()
        if model is not None:
            pairs = [[query, chunk.content] for chunk in chunks]
            scores = model.predict(pairs)
            scored_chunks = list(zip(chunks, [float(s) for s in scores], strict=False))
            scored_chunks.sort(key=lambda x: x[1], reverse=True)
            return scored_chunks[:top_k]

        # Pass-through if model is not yet loaded
        return [(chunk, 1.0 / (idx + 1)) for idx, chunk in enumerate(chunks[:top_k])]
