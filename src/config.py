"""Centralized configuration management for the Evidence-Backed RAG System."""

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent

INSUFFICIENT_EVIDENCE_REFUSAL = (
    "Insufficient evidence to answer this question based on the provided documents."
)


class ChunkingConfig(BaseSettings):
    """Configuration for chunking strategies."""
    tokenizer_encoding: str = "cl100k_base"

    # Strategy A: Baseline Fixed-Size Token Chunking
    strategy_a_chunk_size: int = 500
    strategy_a_chunk_overlap: int = 50

    # Strategy B: Fine-Grained 200-Token Chunking with Sentence-Boundary Preservation
    strategy_b_chunk_size: int = 200
    strategy_b_chunk_overlap: int = 20
    strategy_b_preserve_sentences: bool = True


class RAGConfig(BaseSettings):
    """Main application and RAG pipeline settings."""
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Environment & Paths
    base_dir: Path = BASE_DIR
    raw_data_dir: Path = BASE_DIR / "data" / "raw"
    processed_data_dir: Path = BASE_DIR / "data" / "processed"
    golden_qa_path: Path = BASE_DIR / "data" / "golden_qa" / "qa_pairs.jsonl"
    eval_results_dir: Path = BASE_DIR / "evaluation" / "results"

    # Vector DB & Embeddings
    chroma_persist_dir: str = str(BASE_DIR / "chroma_db")
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Retrieval Configuration
    retrieval_strategy: Literal["dense_only", "bm25_only", "hybrid", "hybrid_rerank"] = (
        "hybrid_rerank"
    )
    dense_top_k: int = 10
    sparse_top_k: int = 10
    final_top_k: int = 5
    rrf_k: int = 60

    # Cross-Encoder Reranker
    reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_top_k: int = 20
    candidate_top_k: int = 20

    # Generation & Refusal
    active_chunking_strategy: Literal["strategy_a", "strategy_b"] = "strategy_a"
    evidence_confidence_threshold: float = 0.65
    refusal_message: str = INSUFFICIENT_EVIDENCE_REFUSAL

    # LLM Settings
    openai_api_key: str = Field(default="", validation_alias="OPENAI_API_KEY")
    anthropic_api_key: str = Field(default="", validation_alias="ANTHROPIC_API_KEY")
    ollama_base_url: str = Field(default="http://localhost:11434", validation_alias="OLLAMA_BASE_URL")
    llm_model_name: str = "gpt-4o-mini"
    llm_temperature: float = 0.0

    # API & Serving
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    streamlit_port: int = 8501

    chunking: ChunkingConfig = ChunkingConfig()


settings = RAGConfig()
