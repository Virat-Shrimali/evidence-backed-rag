"""Centralized configuration management for the Evidence-Backed RAG System."""

from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field
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
    llm_provider: Literal["mock", "openai", "anthropic", "ollama"] = Field(
        default="mock",
        validation_alias=AliasChoices("LLM_PROVIDER", "llm_provider"),
    )
    openai_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("OPENAI_API_KEY", "openai_api_key"),
    )
    anthropic_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("ANTHROPIC_API_KEY", "anthropic_api_key"),
    )
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        validation_alias=AliasChoices("OLLAMA_BASE_URL", "ollama_base_url"),
    )
    llm_model_name: str = Field(
        default="gpt-4o-mini",
        validation_alias=AliasChoices("LLM_MODEL_NAME", "llm_model_name"),
    )
    llm_temperature: float = Field(
        default=0.0,
        validation_alias=AliasChoices("LLM_TEMPERATURE", "llm_temperature"),
    )

    # API & Serving
    api_host: str = Field(
        default="0.0.0.0",
        validation_alias=AliasChoices("API_HOST", "api_host"),
    )
    api_port: int = Field(
        default=8000,
        validation_alias=AliasChoices("PORT", "API_PORT", "api_port"),
    )
    streamlit_port: int = Field(
        default=8501,
        validation_alias=AliasChoices("STREAMLIT_PORT", "streamlit_port"),
    )

    chunking: ChunkingConfig = ChunkingConfig()


settings = RAGConfig()
