"""Unit tests for generation parsing, citations, evidence validation, and refusal enforcement."""

import os

import pytest

from src.config import INSUFFICIENT_EVIDENCE_REFUSAL
from src.generation.generate import (
    Citation,
    EvidenceGroundedGenerator,
    parse_llm_json_response,
    validate_citations,
)
from src.generation.providers import (
    MockLLMProvider,
    OpenAILLMProvider,
    create_llm_provider,
)
from src.pipeline import RAGPipeline
from src.retrieval.models import RetrievedChunk
from src.retrieval.retriever import BaseRetriever


@pytest.fixture
def sample_retrieved_chunks() -> list[RetrievedChunk]:
    """Sample retrieved chunks with verifiable content and scores."""
    return [
        RetrievedChunk(
            chunk_id="doc1#stratA#c0001",
            document_id="doc1",
            content="Operating margin increased by 3.5% due to automation efficiencies in 2023.",
            score=0.88,
            rank=1,
            retrieval_method="dense",
            page_numbers=[1],
        ),
        RetrievedChunk(
            chunk_id="doc1#stratA#c0002",
            document_id="doc1",
            content="Total annual revenue reached $120 million across all primary operating business units.",
            score=0.82,
            rank=2,
            retrieval_method="dense",
            page_numbers=[2],
        ),
    ]


def test_citation_model_aliases():
    """Verify Citation supports both text_snippet and snippet alias."""
    c1 = Citation(chunk_id="doc1#c1", text_snippet="revenue was $120M")
    assert c1.text_snippet == "revenue was $120M"
    assert c1.snippet == "revenue was $120M"

    c2 = Citation(chunk_id="doc1#c2", snippet="margin increased 3.5%")
    assert c2.text_snippet == "margin increased 3.5%"
    assert c2.snippet == "margin increased 3.5%"


def test_parse_valid_llm_response():
    """Verify structured response with citations is correctly parsed."""
    raw_json = """
    {
        "answer": "Revenue reached 120 million in FY2023.",
        "citations": [{"chunk_id": "doc1#chunk-0", "snippet": "revenue of $120 million"}],
        "sufficient_evidence": true,
        "confidence": 0.95
    }
    """
    response = parse_llm_json_response(raw_json, retrieved_chunk_ids=["doc1#chunk-0"])
    assert response.sufficient_evidence is True
    assert response.confidence == 0.95
    assert len(response.citations) == 1
    assert response.citations[0].chunk_id == "doc1#chunk-0"
    assert response.citations[0].snippet == "revenue of $120 million"
    assert response.refusal_reason is None


def test_refusal_on_insufficient_evidence():
    """Verify system strictly returns standard refusal message when evidence is insufficient."""
    raw_json = """
    {
        "answer": "I am not sure about the quarterly dividend rate.",
        "citations": [],
        "sufficient_evidence": false,
        "confidence": 0.1,
        "refusal_reason": "No dividend information mentioned"
    }
    """
    response = parse_llm_json_response(raw_json, retrieved_chunk_ids=["doc1#chunk-0"])
    assert response.sufficient_evidence is False
    assert response.answer == INSUFFICIENT_EVIDENCE_REFUSAL
    assert response.confidence == 0.0


def test_refusal_on_low_confidence():
    """Verify refusal triggers if confidence is below threshold even if marked sufficient."""
    raw_json = """
    {
        "answer": "Maybe it was 10 percent.",
        "citations": [{"chunk_id": "doc1#chunk-0", "snippet": "10 percent"}],
        "sufficient_evidence": true,
        "confidence": 0.40
    }
    """
    response = parse_llm_json_response(raw_json, retrieved_chunk_ids=["doc1#chunk-0"])
    assert response.sufficient_evidence is False
    assert response.answer == INSUFFICIENT_EVIDENCE_REFUSAL


def test_refusal_on_malformed_json():
    """Verify malformed JSON gracefully triggers standard refusal instead of crashing."""
    malformed = "This is not valid JSON at all."
    response = parse_llm_json_response(malformed, retrieved_chunk_ids=["doc1#chunk-0"])
    assert response.sufficient_evidence is False
    assert response.answer == INSUFFICIENT_EVIDENCE_REFUSAL


def test_validate_citations_success(sample_retrieved_chunks):
    """Verify valid citation with verifiable snippet passes validation."""
    citations = [
        Citation(chunk_id="doc1#stratA#c0001", text_snippet="margin increased by 3.5%"),
        Citation(chunk_id="doc1#stratA#c0002", text_snippet="annual revenue reached $120 million"),
    ]
    is_valid, validated, err = validate_citations(citations, retrieved_chunks=sample_retrieved_chunks)
    assert is_valid is True
    assert len(validated) == 2
    assert err is None


def test_validate_citations_rejects_nonexistent_chunk(sample_retrieved_chunks):
    """Verify citing a chunk_id not present in retrieved context fails validation."""
    citations = [
        Citation(chunk_id="doc999#hallucinated", text_snippet="fake snippet"),
    ]
    is_valid, validated, err = validate_citations(citations, retrieved_chunks=sample_retrieved_chunks)
    assert is_valid is False
    assert len(validated) == 0
    assert "non-existent chunk ID" in err


def test_validate_citations_rejects_fabricated_snippet(sample_retrieved_chunks):
    """Verify citing a real chunk ID with a fabricated text snippet fails validation."""
    citations = [
        Citation(chunk_id="doc1#stratA#c0001", text_snippet="operating expenses tripled overnight"),
    ]
    is_valid, validated, err = validate_citations(citations, retrieved_chunks=sample_retrieved_chunks)
    assert is_valid is False
    assert len(validated) == 0
    assert "not found in cited chunk" in err


def test_validate_citations_deduplication(sample_retrieved_chunks):
    """Verify identical duplicate citations are deduplicated."""
    citations = [
        Citation(chunk_id="doc1#stratA#c0001", text_snippet="margin increased by 3.5%"),
        Citation(chunk_id="doc1#stratA#c0001", text_snippet="margin increased by 3.5%"),
    ]
    is_valid, validated, err = validate_citations(citations, retrieved_chunks=sample_retrieved_chunks)
    assert is_valid is True
    assert len(validated) == 1


def test_generator_deterministic_refusal_on_empty_chunks():
    """Verify generator returns standard refusal immediately when retrieved chunks is empty."""
    generator = EvidenceGroundedGenerator(provider=MockLLMProvider())
    response = generator.generate("What was the revenue?", retrieved_chunks=[])
    assert response.sufficient_evidence is False
    assert response.answer == INSUFFICIENT_EVIDENCE_REFUSAL
    assert response.confidence == 0.0
    assert response.citations == []
    assert "No retrieved context" in (response.refusal_reason or "")


def test_generator_deterministic_refusal_on_low_score_chunks():
    """Verify generator refuses when retrieved chunks have score below confidence threshold."""
    low_score_chunks = [
        RetrievedChunk(
            chunk_id="c_low",
            document_id="doc1",
            content="Unrelated text content.",
            score=0.25,  # Below 0.65 threshold
            rank=1,
            retrieval_method="dense",
        )
    ]
    generator = EvidenceGroundedGenerator(
        provider=MockLLMProvider(),
        confidence_threshold=0.65,
    )
    response = generator.generate("What was the revenue?", retrieved_chunks=low_score_chunks)
    assert response.sufficient_evidence is False
    assert response.answer == INSUFFICIENT_EVIDENCE_REFUSAL
    assert response.confidence == 0.0
    assert "below threshold" in (response.refusal_reason or "")


def test_generator_successful_grounded_answer(sample_retrieved_chunks):
    """Verify generator produces structured response with valid citations and provenance."""
    canned = """
    {
        "answer": "Annual revenue reached $120 million in FY2023.",
        "citations": [{"chunk_id": "doc1#stratA#c0002", "text_snippet": "revenue reached $120 million"}],
        "sufficient_evidence": true,
        "confidence": 0.95,
        "refusal_reason": null
    }
    """
    mock_provider = MockLLMProvider(canned_response=canned)
    generator = EvidenceGroundedGenerator(provider=mock_provider)

    response = generator.generate("What was the total revenue?", retrieved_chunks=sample_retrieved_chunks)

    assert response.sufficient_evidence is True
    assert response.answer == "Annual revenue reached $120 million in FY2023."
    assert len(response.citations) == 1
    assert response.citations[0].chunk_id == "doc1#stratA#c0002"
    assert response.citations[0].text_snippet == "revenue reached $120 million"
    assert "doc1#stratA#c0001" in response.retrieved_chunk_ids
    assert "doc1#stratA#c0002" in response.retrieved_chunk_ids


def test_generator_refuses_when_llm_hallucinates_chunk_id(sample_retrieved_chunks):
    """Verify generator catches hallucinated chunk ID and triggers refusal."""
    hallucinated_output = """
    {
        "answer": "Revenue reached 120 million.",
        "citations": [{"chunk_id": "fake_doc#chunk-999", "text_snippet": "revenue reached $120 million"}],
        "sufficient_evidence": true,
        "confidence": 0.95
    }
    """
    generator = EvidenceGroundedGenerator(provider=MockLLMProvider(canned_response=hallucinated_output))
    response = generator.generate("What was the revenue?", retrieved_chunks=sample_retrieved_chunks)

    assert response.sufficient_evidence is False
    assert response.answer == INSUFFICIENT_EVIDENCE_REFUSAL
    assert response.citations == []
    assert "non-existent chunk ID" in (response.refusal_reason or "")


def test_generator_refuses_when_llm_fabricates_snippet(sample_retrieved_chunks):
    """Verify generator catches fabricated snippet and triggers refusal."""
    fabricated_output = """
    {
        "answer": "Revenue was 120 million.",
        "citations": [{"chunk_id": "doc1#stratA#c0001", "text_snippet": "completely fabricated snippet not in chunk"}],
        "sufficient_evidence": true,
        "confidence": 0.95
    }
    """
    generator = EvidenceGroundedGenerator(provider=MockLLMProvider(canned_response=fabricated_output))
    response = generator.generate("What was the revenue?", retrieved_chunks=sample_retrieved_chunks)

    assert response.sufficient_evidence is False
    assert response.answer == INSUFFICIENT_EVIDENCE_REFUSAL
    assert response.citations == []
    assert "not found in cited chunk" in (response.refusal_reason or "")


def test_generator_refuses_when_marked_sufficient_without_citations(sample_retrieved_chunks):
    """Verify generator rejects an answer claiming sufficient evidence but providing no citations."""
    no_citations_output = """
    {
        "answer": "Revenue reached 120 million based on my knowledge.",
        "citations": [],
        "sufficient_evidence": true,
        "confidence": 0.95
    }
    """
    generator = EvidenceGroundedGenerator(provider=MockLLMProvider(canned_response=no_citations_output))
    response = generator.generate("What was the revenue?", retrieved_chunks=sample_retrieved_chunks)

    assert response.sufficient_evidence is False
    assert response.answer == INSUFFICIENT_EVIDENCE_REFUSAL
    assert response.citations == []
    assert "no citations" in (response.refusal_reason or "")


def test_mock_llm_provider_custom_generator():
    """Verify MockLLMProvider generator_fn custom behavior and call history."""
    def custom_gen(prompt: str, sys: str) -> str:
        return """{"answer": "Custom response", "citations": [], "sufficient_evidence": false, "confidence": 0.0}"""

    provider = MockLLMProvider(generator_fn=custom_gen)
    out = provider.generate_raw("test prompt", "test sys")
    assert "Custom response" in out
    assert len(provider.call_history) == 1
    assert provider.call_history[0]["prompt"] == "test prompt"


def test_provider_factory():
    """Verify create_llm_provider returns requested provider types and raises on invalid."""
    mock = create_llm_provider("mock")
    assert isinstance(mock, MockLLMProvider)

    with pytest.raises(ValueError, match="Unknown LLM provider"):
        create_llm_provider("invalid_provider")


class MockPipelineRetriever(BaseRetriever):
    def __init__(self, chunks: list[RetrievedChunk]):
        self.chunks = chunks

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        return self.chunks[:top_k] if top_k else self.chunks


def test_rag_pipeline_end_to_end(sample_retrieved_chunks):
    """Verify end-to-end RAGPipeline connects retrieval and generation."""
    canned = """
    {
        "answer": "Operating margin expanded by 3.5% due to automation.",
        "citations": [{"chunk_id": "doc1#stratA#c0001", "text_snippet": "margin increased by 3.5%"}],
        "sufficient_evidence": true,
        "confidence": 0.9
    }
    """
    retriever = MockPipelineRetriever(sample_retrieved_chunks)
    generator = EvidenceGroundedGenerator(provider=MockLLMProvider(canned_response=canned))
    pipeline = RAGPipeline(retriever=retriever, generator=generator)

    response = pipeline.query("What happened to the operating margin?")
    assert response.sufficient_evidence is True
    assert "3.5%" in response.answer
    assert len(response.citations) == 1
    assert response.citations[0].chunk_id == "doc1#stratA#c0001"


def test_mock_llm_provider_does_not_refuse_when_context_contains_unanswerable():
    """Verify MockLLMProvider does not trigger false refusal when context mentions 'unanswerable'."""
    chunks_with_unanswerable = [
        RetrievedChunk(
            chunk_id="chunk_eval_guide",
            document_id="doc_guide",
            content="Include a few unanswerable questions in your evaluation dataset to test refusal accuracy.",
            score=0.9,
            rank=1,
            retrieval_method="bm25",
        ),
        RetrievedChunk(
            chunk_id="chunk_metric_data",
            document_id="doc_guide",
            content="Operating margin increased by 4.2% across business units in FY2023.",
            score=0.85,
            rank=2,
            retrieval_method="bm25",
        ),
    ]
    generator = EvidenceGroundedGenerator(provider=MockLLMProvider())
    response = generator.generate("What was the operating margin increase?", retrieved_chunks=chunks_with_unanswerable)

    assert response.sufficient_evidence is True
    assert response.answer != INSUFFICIENT_EVIDENCE_REFUSAL
    assert len(response.citations) == 1
    assert response.citations[0].chunk_id in ("chunk_eval_guide", "chunk_metric_data")


def test_mock_llm_provider_refuses_when_question_is_unanswerable(sample_retrieved_chunks):
    """Verify MockLLMProvider deterministically refuses when the question itself triggers refusal."""
    generator = EvidenceGroundedGenerator(provider=MockLLMProvider())

    for unanswerable_q in [
        "Explain the unanswerable details of this proposal.",
        "How do I set up a redis caching layer?",
        "What is the elasticsearch cluster topology?",
        "Where is the postgresql connection configured?",
    ]:
        response = generator.generate(unanswerable_q, retrieved_chunks=sample_retrieved_chunks)
        assert response.sufficient_evidence is False
        assert response.answer == INSUFFICIENT_EVIDENCE_REFUSAL
        assert response.citations == []


def test_bm25_only_query_retrieves_and_cites_embedding_model():
    """Verify test query 'What embedding model does the project use?' succeeds with bm25_only and cites the embedding model."""
    generator = EvidenceGroundedGenerator(provider=MockLLMProvider())
    pipeline = RAGPipeline(generator=generator)

    response = pipeline.query("What embedding model does the project use?", retriever_mode="bm25_only")
    assert response.sufficient_evidence is True
    assert response.answer != INSUFFICIENT_EVIDENCE_REFUSAL
    assert len(response.citations) >= 1

    cited_text = " ".join(c.text_snippet for c in response.citations)
    assert any(term in cited_text.lower() or term in response.answer.lower() for term in ["minilm", "sentence-transformers"])


@pytest.mark.integration
def test_real_llm_provider_integration():
    """Optional integration test with real OpenAI or Ollama provider if configured in environment."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set; skipping live LLM integration test.")

    provider = OpenAILLMProvider(api_key=api_key)
    raw = provider.generate_raw("Return JSON with answer: 'pong'", "You must respond in JSON format.")
    assert "pong" in raw.lower()


