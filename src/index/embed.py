"""Dense embedding and vector store integration."""


from src.config import settings
from src.ingest.chunk import Chunk


class DenseIndex:
    """Dense vector store interface using ChromaDB and SentenceTransformers."""

    def __init__(
        self,
        persist_dir: str | None = None,
        model_name: str | None = None,
    ):
        self.persist_dir = persist_dir or settings.chroma_persist_dir
        self.model_name = model_name or settings.embedding_model_name
        self.client = None
        self.collection = None

    def index_chunks(self, chunks: list[Chunk], collection_name: str = "rag_chunks") -> None:
        """Add chunks to Chroma vector collection."""
        if not chunks:
            return

        try:
            import chromadb
            from chromadb.utils import embedding_functions

            self.client = chromadb.PersistentClient(path=self.persist_dir)
            emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self.model_name
            )
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                embedding_function=emb_fn,
            )

            ids = [c.id for c in chunks]
            documents = [c.content for c in chunks]
            metadatas = [c.metadata for c in chunks]

            self.collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
            )
        except ImportError:
            # Stored in-memory fallback for early scaffolding
            self._in_memory_chunks = {c.id: c for c in chunks}

    def search(self, query: str, top_k: int = 5, collection_name: str = "rag_chunks") -> list[str]:
        """Search dense index and return ranked chunk IDs."""
        if self.collection is not None:
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k,
            )
            if results and results.get("ids") and results["ids"][0]:
                return results["ids"][0]
        return []
