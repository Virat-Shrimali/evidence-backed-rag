"""Dense embedding and vector store indexing using SentenceTransformers and ChromaDB."""

import json
from pathlib import Path
from typing import Any

import chromadb
from chromadb.utils import embedding_functions

from src.config import settings
from src.ingest.chunk import Chunk
from src.retrieval.models import RetrievedChunk


class DenseIndex:
    """Persistent vector store interface for dense indexing and retrieval."""

    def __init__(
        self,
        persist_dir: str | Path | None = None,
        model_name: str | None = None,
    ):
        self.persist_dir = str(persist_dir or settings.chroma_persist_dir)
        self.model_name = model_name or settings.embedding_model_name
        self._client: chromadb.PersistentClient | None = None
        self._embedding_fn = None

    def _get_client(self) -> chromadb.PersistentClient:
        """Lazily initialize and return persistent ChromaDB client."""
        if self._client is None:
            Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=self.persist_dir)
        return self._client

    def _get_embedding_fn(self):
        """Lazily load sentence-transformers embedding function."""
        if self._embedding_fn is None:
            self._embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self.model_name
            )
        return self._embedding_fn

    def get_collection(
        self, collection_name: str = "rag_chunks"
    ) -> chromadb.Collection:
        """Get or create a Chroma collection configured with cosine space."""
        client = self._get_client()
        return client.get_or_create_collection(
            name=collection_name,
            embedding_function=self._get_embedding_fn(),
            metadata={"hnsw:space": "cosine"},
        )

    def index_chunks(
        self,
        chunks: list[Chunk],
        collection_name: str = "rag_chunks",
    ) -> int:
        """Index chunks into persistent vector collection.

        Preserves complete chunk provenance and metadata.
        Returns the number of indexed chunks.
        """
        if not chunks:
            return 0

        # Deduplicate chunks by ID (preserving the latest occurrence)
        unique_chunks_map: dict[str, Chunk] = {c.id: c for c in chunks}
        unique_chunks = list(unique_chunks_map.values())

        collection = self.get_collection(collection_name)

        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict[str, Any]] = []

        for chunk in unique_chunks:
            ids.append(chunk.id)
            documents.append(chunk.content)

            # Chroma requires metadata values to be str, int, float, or bool
            flat_meta: dict[str, Any] = {
                "document_id": chunk.document_id,
                "chunk_index": chunk.chunk_index,
                "token_count": chunk.token_count,
                "page_numbers_json": json.dumps(chunk.page_numbers),
                "metadata_json": json.dumps(chunk.metadata),
            }

            # Copy primitive custom metadata fields to top-level
            for k, v in chunk.metadata.items():
                if isinstance(v, (str, int, float, bool)):
                    flat_meta[f"meta_{k}"] = v

            metadatas.append(flat_meta)

        # Batch upsert into Chroma
        collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )
        return len(unique_chunks)

    def search(
        self,
        query: str,
        top_k: int = 5,
        collection_name: str = "rag_chunks",
    ) -> list[RetrievedChunk]:
        """Query dense vector index and return ranked chunks with provenance."""
        if not query.strip():
            return []

        collection = self.get_collection(collection_name)
        total_docs = collection.count()
        if total_docs == 0:
            return []

        n_results = min(max(1, top_k), total_docs)
        results = collection.query(
            query_texts=[query],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

        if not results or not results.get("ids") or not results["ids"][0]:
            return []

        retrieved: list[RetrievedChunk] = []
        ids = results["ids"][0]
        documents = results["documents"][0] if results.get("documents") else []
        metadatas = results["metadatas"][0] if results.get("metadatas") else []
        distances = results["distances"][0] if results.get("distances") else []

        for rank, chunk_id in enumerate(ids):
            content = documents[rank] if rank < len(documents) else ""
            meta_dict = metadatas[rank] if rank < len(metadatas) else {}
            distance = distances[rank] if rank < len(distances) else 1.0

            # Cosine distance in Chroma is 1 - cosine_similarity
            similarity_score = float(max(0.0, 1.0 - distance))

            # Reconstruct original page numbers and metadata dict
            pages_json = meta_dict.get("page_numbers_json", "[]")
            try:
                page_numbers = json.loads(pages_json)
            except Exception:
                page_numbers = []

            raw_meta_json = meta_dict.get("metadata_json", "{}")
            try:
                orig_metadata = json.loads(raw_meta_json)
            except Exception:
                orig_metadata = {}

            doc_id = str(meta_dict.get("document_id", ""))

            retrieved.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    document_id=doc_id,
                    content=content,
                    score=similarity_score,
                    rank=rank + 1,
                    retrieval_method="dense",
                    page_numbers=page_numbers,
                    metadata=orig_metadata,
                )
            )

        return retrieved

    def count(self, collection_name: str = "rag_chunks") -> int:
        """Return the count of chunks in the collection."""
        try:
            return self.get_collection(collection_name).count()
        except Exception:
            return 0

    def delete_collection(self, collection_name: str = "rag_chunks") -> None:
        """Delete collection from persistent Chroma store."""
        try:
            self._get_client().delete_collection(name=collection_name)
        except Exception:
            pass
