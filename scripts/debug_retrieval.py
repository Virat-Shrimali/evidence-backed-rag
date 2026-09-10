"""Diagnostic script to inspect BM25 retrieval and MockLLMProvider behavior."""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import settings  # noqa: E402
from src.generation.generate import EvidenceGroundedGenerator  # noqa: E402
from src.generation.providers import MockLLMProvider  # noqa: E402
from src.pipeline import RAGPipeline  # noqa: E402
from src.retrieval.retriever import create_retriever  # noqa: E402


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    query = "What embedding model does the project use?"
    print("=== DIAGNOSTIC RETRIEVAL RUN ===")
    print(f"Query: '{query}'")
    print("Retriever mode: bm25_only")
    print("Top-k: 5\n")

    # 1. Create BM25 retriever
    retriever = create_retriever(strategy="bm25_only", config=settings)
    chunks = retriever.retrieve(query, top_k=5)

    print(f"Retrieved {len(chunks)} chunks:\n" + "=" * 60)
    for _i, c in enumerate(chunks, start=1):
        print(f"--- [Rank {c.rank}] Chunk ID: {c.chunk_id} ---")
        print(f"Score: {c.score}")
        print(f"Document ID: {c.document_id}")
        print(f"Page Numbers: {c.page_numbers}")
        print(f"Retrieval Method: {c.retrieval_method}")
        print(f"Full Content:\n{c.content}\n" + "-" * 60)

    # 2. Check if embedding model is mentioned in the retrieved chunks
    target_terms = ["embedding", "minilm", "sentence-transformers", "all-minilm-l6-v2"]
    print("\nTarget Term Occurrences in Retrieved Chunks:")
    found_any = False
    for c in chunks:
        found_terms = [t for t in target_terms if t in c.content.lower()]
        if found_terms:
            found_any = True
            print(f"  Chunk {c.chunk_id} mentions: {found_terms}")
    if not found_any:
        print("  NO target terms found in any retrieved chunk!")

    # 3. Test generation with default MockLLMProvider
    print("\n" + "=" * 60)
    print("Testing EvidenceGroundedGenerator with MockLLMProvider:")
    mock_provider = MockLLMProvider()
    generator = EvidenceGroundedGenerator(provider=mock_provider, config=settings)
    rag_response = generator.generate(query, chunks)

    print(f"Answer: {rag_response.answer}")
    print(f"Sufficient Evidence: {rag_response.sufficient_evidence}")
    print(f"Confidence: {rag_response.confidence}")
    print(f"Citations: {rag_response.citations}")
    print(f"Refusal Reason: {rag_response.refusal_reason}")

    # Also test through pipeline
    print("\n" + "=" * 60)
    print("Testing through RAGPipeline(retriever_mode='bm25_only'):")
    pipeline = RAGPipeline(generator=generator)
    pipe_response = pipeline.query(query, retriever_mode="bm25_only")
    print(f"Pipeline Answer: {pipe_response.answer}")
    print(f"Pipeline Sufficient Evidence: {pipe_response.sufficient_evidence}")
    print(f"Pipeline Citations: {pipe_response.citations}")
    print(f"Pipeline Refusal Reason: {pipe_response.refusal_reason}")


if __name__ == "__main__":
    main()
