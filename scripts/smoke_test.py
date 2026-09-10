"""End-to-end smoke test script validating the complete RAG pipeline locally."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def run_smoke_test() -> bool:
    print("=== Starting Evidence-Backed RAG End-to-End Smoke Test ===")

    from fastapi.testclient import TestClient

    from api.main import app, get_pipeline
    from src.config import settings
    from src.generation.generate import EvidenceGroundedGenerator
    from src.generation.providers import MockLLMProvider
    from src.index.bm25_index import BM25Index
    from src.index.embed import DenseIndex
    from src.ingest.chunk import chunk_document
    from src.ingest.parse import parse_directory
    from src.pipeline import RAGPipeline
    from src.retrieval.rerank import CrossEncoderReranker
    from src.retrieval.retriever import (
        BM25Retriever,
        DenseRetriever,
        HybridRetriever,
        RerankedRetriever,
    )

    # 1. Parse documents
    raw_dir = settings.raw_data_dir
    print(f"[1/11] Parsing documents from {raw_dir}...")
    parsed_docs = parse_directory(raw_dir)
    assert len(parsed_docs) > 0, "Failed: No documents parsed!"
    doc = parsed_docs[0]
    print(f"       Parsed: '{doc.source}' ({len(doc.content)} chars, {len(doc.pages)} pages)")

    # 2. Chunk documents
    print("[2/11] Generating chunks with Strategy A...")
    chunks = chunk_document(doc, strategy="strategy_a")
    assert len(chunks) > 0, "Failed: No chunks produced!"
    print(f"       Generated {len(chunks)} chunks. First chunk ID: {chunks[0].id}")

    # 3. Build/load indexes
    print("[3/11] Building BM25 index & loading Dense index...")
    bm25_idx = BM25Index(chunks)
    assert len(bm25_idx) == len(chunks), "BM25 index chunk count mismatch!"

    dense_idx = DenseIndex(persist_dir=settings.chroma_persist_dir)
    collection = dense_idx.get_collection(collection_name="rag_chunks")
    if collection.count() == 0:
        print("       Populating dense index with corpus chunks...")
        dense_idx.index_chunks(chunks)
    print(f"       Dense index ready. Collection count: {collection.count()}")

    # 4. Retrieval strategies (BM25 and Dense)
    test_query = "What is the core differentiator of the Evidence-Backed RAG system?"
    print(f"[4/11] Testing BM25 & Dense retrieval for: '{test_query[:40]}...'")
    bm25_retriever = BM25Retriever(bm25_index=bm25_idx, top_k=5)
    bm25_results = bm25_retriever.retrieve(test_query)
    assert len(bm25_results) > 0, "BM25 retrieval returned empty!"
    print(f"       BM25 returned {len(bm25_results)} chunks (top score: {bm25_results[0].score:.3f})")

    dense_retriever = DenseRetriever(dense_index=dense_idx, top_k=5)
    dense_results = dense_retriever.retrieve(test_query)
    assert len(dense_results) > 0, "Dense retrieval returned empty!"
    print(f"       Dense returned {len(dense_results)} chunks (top score: {dense_results[0].score:.3f})")

    # 5. Hybrid retrieval (RRF)
    print("[5/11] Testing Hybrid retrieval (Reciprocal Rank Fusion)...")
    hybrid_retriever = HybridRetriever(
        dense_retriever=dense_retriever,
        bm25_retriever=bm25_retriever,
        final_top_k=5,
    )
    hybrid_results = hybrid_retriever.retrieve(test_query)
    assert len(hybrid_results) > 0, "Hybrid retrieval returned empty!"
    print(f"       Hybrid returned {len(hybrid_results)} chunks (top score: {hybrid_results[0].score:.4f})")

    # 6. Cross-Encoder reranking
    print("[6/11] Testing Cross-Encoder reranking...")
    reranker = CrossEncoderReranker(model_name=settings.reranker_model_name)
    reranked_retriever = RerankedRetriever(base_retriever=hybrid_retriever, reranker=reranker, final_top_k=5)
    reranked_results = reranked_retriever.retrieve(test_query)
    assert len(reranked_results) > 0, "Reranked retrieval returned empty!"
    top_chunk = reranked_results[0]
    print(f"       Reranked returned {len(reranked_results)} chunks (top ID: {top_chunk.chunk_id}, score: {top_chunk.score:.3f})")

    # 7. Grounded generation with citation
    print("[7/11] Testing grounded generation with citations (Mock LLM)...")
    words = top_chunk.content.split()
    snippet = " ".join(words[:6])
    mock_payload = {
        "answer": f"Based on the retrieved documentation: {snippet}.",
        "citations": [{"chunk_id": top_chunk.chunk_id, "text_snippet": snippet}],
        "confidence": 0.95,
        "sufficient_evidence": True,
    }
    mock_llm = MockLLMProvider(canned_response=json.dumps(mock_payload))
    generator = EvidenceGroundedGenerator(provider=mock_llm, config=settings)
    rag_response = generator.generate(test_query, reranked_results)
    assert rag_response.sufficient_evidence is True, "Expected sufficient evidence!"
    assert len(rag_response.citations) > 0, "Expected at least one citation!"
    print(f"       Generated answer: '{rag_response.answer[:60]}...'")
    print(f"       Citations: {len(rag_response.citations)} (ID: {rag_response.citations[0].chunk_id})")

    # 8. Cited chunk IDs actually exist
    print("[8/11] Verifying cited chunk IDs exist in candidate chunks...")
    retrieved_ids = {c.chunk_id for c in reranked_results}
    for cit in rag_response.citations:
        assert cit.chunk_id in retrieved_ids, f"Cited chunk {cit.chunk_id} not in retrieved chunks!"
    print("       All cited chunk IDs confirmed present in retrieved candidates.")

    # 9. Citation snippets occur in cited chunks
    print("[9/11] Verifying citation text snippets occur in chunk contents...")
    chunk_dict = {c.chunk_id: c.content for c in reranked_results}
    for cit in rag_response.citations:
        chunk_content = chunk_dict[cit.chunk_id]
        assert cit.text_snippet.lower() in chunk_content.lower(), f"Snippet '{cit.text_snippet}' not found in chunk content!"
    print("       All citation text snippets verified against chunk text.")

    # 10. Deterministic refusal on unanswerable question
    print("[10/11] Testing deterministic refusal on unanswerable question...")
    unanswerable_query = "What is the secret recipe for Martian hot chocolate?"
    # Pass empty candidates to trigger deterministic refusal before LLM
    refusal_response = generator.generate(unanswerable_query, [])
    assert refusal_response.sufficient_evidence is False, "Refusal expected sufficient_evidence=False!"
    assert refusal_response.answer == settings.refusal_message, "Answer did not match exact standard refusal message!"
    print(f"       Correct refusal returned: '{refusal_response.answer}'")

    # 11. FastAPI /query returns structured response
    print("[11/11] Testing FastAPI POST /query endpoint...")
    client = TestClient(app)
    # Override pipeline to use our verified pipeline with mock generator
    app.dependency_overrides[get_pipeline] = lambda: RAGPipeline(
        retriever=reranked_retriever,
        generator=generator,
    )
    api_res = client.post("/query", json={"question": test_query})
    assert api_res.status_code == 200, f"API query returned {api_res.status_code}!"
    body = api_res.json()
    assert body["sufficient_evidence"] is True
    assert len(body["citations"]) > 0
    assert body["citations"][0]["chunk_id"] == top_chunk.chunk_id
    print("       FastAPI endpoint verified with HTTP 200 and valid structured JSON schema.")

    app.dependency_overrides.clear()
    print("\n>>> ALL 11 END-TO-END SMOKE TEST CHECKS PASSED SUCCESSFULLY! <<<")
    return True


if __name__ == "__main__":
    success = run_smoke_test()
    sys.exit(0 if success else 1)
