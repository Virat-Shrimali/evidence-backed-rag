"""Schema and utilities to load, save, and validate golden QA evaluation pairs."""

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class GoldenQAPair(BaseModel):
    """Schema for a golden evaluation pair."""

    id: str
    question: str
    expected_answer: str
    relevant_chunk_ids: list[str] = Field(default_factory=list)
    is_answerable: bool = True
    difficulty: str = "easy"  # easy, medium, multi_hop, unanswerable
    category: str = "general"


def load_golden_qa_set(file_path: Path | str) -> list[GoldenQAPair]:
    """Load and validate golden QA pairs from a JSONL file."""
    path = Path(file_path)
    if not path.exists():
        return []

    pairs: list[GoldenQAPair] = []
    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                pairs.append(GoldenQAPair.model_validate_json(line))
            except Exception as e:
                raise ValueError(
                    f"Error parsing golden QA pair at {path}:{line_num}: {e}"
                ) from e
    return pairs


def save_golden_qa_set(pairs: list[GoldenQAPair], file_path: Path | str) -> None:
    """Save golden QA pairs to a JSONL file."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(pair.model_dump_json() + "\n")


def validate_golden_qa_set(pairs: list[GoldenQAPair]) -> dict[str, Any]:
    """Validate consistency of golden QA dataset and return summary statistics."""
    if not pairs:
        raise ValueError("Golden QA dataset is empty")

    seen_ids: set[str] = set()
    category_counts: dict[str, int] = {}
    difficulty_counts: dict[str, int] = {}
    answerable_count = 0
    unanswerable_count = 0

    for pair in pairs:
        if not pair.id:
            raise ValueError(f"Golden QA pair missing ID: {pair}")
        if pair.id in seen_ids:
            raise ValueError(f"Duplicate Golden QA ID found: {pair.id}")
        seen_ids.add(pair.id)

        if not pair.question.strip():
            raise ValueError(f"Empty question in pair {pair.id}")
        if not pair.expected_answer.strip():
            raise ValueError(f"Empty expected answer in pair {pair.id}")

        if pair.is_answerable:
            answerable_count += 1
            if not pair.relevant_chunk_ids:
                raise ValueError(
                    f"Answerable pair {pair.id} has empty relevant_chunk_ids"
                )
        else:
            unanswerable_count += 1
            if pair.relevant_chunk_ids:
                raise ValueError(
                    f"Unanswerable pair {pair.id} should not have relevant_chunk_ids"
                )

        category_counts[pair.category] = category_counts.get(pair.category, 0) + 1
        difficulty_counts[pair.difficulty] = difficulty_counts.get(pair.difficulty, 0) + 1

    return {
        "total_pairs": len(pairs),
        "answerable_count": answerable_count,
        "unanswerable_count": unanswerable_count,
        "categories": category_counts,
        "difficulties": difficulty_counts,
    }


def create_default_golden_dataset(output_path: Path | str | None = None) -> list[GoldenQAPair]:
    """Create the standard curated golden QA pairs grounded in the development corpus."""
    target_path = Path(output_path) if output_path else Path("data/golden_qa/qa_pairs.jsonl")

    pairs = [
        GoldenQAPair(
            id="qa_001",
            question="What is the core differentiator of the Evidence-Backed RAG system compared to standard demo RAG systems?",
            expected_answer="Every answer cites the exact chunk it came from, and the system explicitly responds with 'insufficient evidence to answer' when it cannot ground a claim instead of hallucinating.",
            relevant_chunk_ids=["doc_evidence-backed-rag-project-guide_1a754e0a#stratA#c0000"],
            difficulty="easy",
            is_answerable=True,
            category="overview",
        ),
        GoldenQAPair(
            id="qa_002",
            question="Which embedding models are recommended for local vector generation to avoid API costs during experimentation?",
            expected_answer="sentence-transformers with all-MiniLM-L6-v2 or bge-small-en-v1.5.",
            relevant_chunk_ids=["doc_evidence-backed-rag-project-guide_1a754e0a#stratA#c0000"],
            difficulty="easy",
            is_answerable=True,
            category="tech_stack",
        ),
        GoldenQAPair(
            id="qa_003",
            question="Which vector store options are recommended in the tech stack for zero-infra local vector storage?",
            expected_answer="ChromaDB (local, zero-infra) or FAISS (lighter, no server).",
            relevant_chunk_ids=["doc_evidence-backed-rag-project-guide_1a754e0a#stratA#c0000"],
            difficulty="easy",
            is_answerable=True,
            category="tech_stack",
        ),
        GoldenQAPair(
            id="qa_004",
            question="What specific model is recommended for cross-encoder reranking in the project guide?",
            expected_answer="cross-encoder/ms-marco-MiniLM-L-6-v2 via sentence-transformers.",
            relevant_chunk_ids=["doc_evidence-backed-rag-project-guide_1a754e0a#stratA#c0001"],
            difficulty="easy",
            is_answerable=True,
            category="reranking",
        ),
        GoldenQAPair(
            id="qa_005",
            question="Why is FastAPI chosen for the backend/API layer according to the tech stack table?",
            expected_answer="FastAPI is chosen because it is an industry standard, supports asynchronous operations, and provides automatic API documentation.",
            relevant_chunk_ids=["doc_evidence-backed-rag-project-guide_1a754e0a#stratA#c0001"],
            difficulty="easy",
            is_answerable=True,
            category="tech_stack",
        ),
        GoldenQAPair(
            id="qa_006",
            question="Which evaluation framework library is suggested for measuring context precision, recall, faithfulness, and answer correctness?",
            expected_answer="The ragas library.",
            relevant_chunk_ids=["doc_evidence-backed-rag-project-guide_1a754e0a#stratA#c0001"],
            difficulty="easy",
            is_answerable=True,
            category="evaluation",
        ),
        GoldenQAPair(
            id="qa_007",
            question="In the architecture workflow, what action is taken if the confidence check determines there is not enough evidence?",
            expected_answer="The system returns a refusal stating 'insufficient evidence to answer' instead of returning an answer.",
            relevant_chunk_ids=["doc_evidence-backed-rag-project-guide_1a754e0a#stratA#c0002"],
            difficulty="easy",
            is_answerable=True,
            category="architecture",
        ),
        GoldenQAPair(
            id="qa_008",
            question="What chunk size and overlap parameters are suggested for the initial baseline in Phase 1 corpus preparation?",
            expected_answer="A fixed chunk size of 500 tokens with 50 tokens of overlap.",
            relevant_chunk_ids=["doc_evidence-backed-rag-project-guide_1a754e0a#stratA#c0002"],
            difficulty="easy",
            is_answerable=True,
            category="chunking",
        ),
        GoldenQAPair(
            id="qa_009",
            question="Under what conditions should the system return an 'insufficient evidence to answer' refusal according to Phase 2 build instructions?",
            expected_answer="If the retrieved chunks' similarity scores are all below a threshold, or if the LLM itself flags low confidence.",
            relevant_chunk_ids=["doc_evidence-backed-rag-project-guide_1a754e0a#stratA#c0003"],
            difficulty="medium",
            is_answerable=True,
            category="refusal",
        ),
        GoldenQAPair(
            id="qa_010",
            question="What structured JSON format must the LLM output during the generation phase?",
            expected_answer="{answer, citations: [{chunk_id, text_snippet}], confidence}",
            relevant_chunk_ids=["doc_evidence-backed-rag-project-guide_1a754e0a#stratA#c0003"],
            difficulty="easy",
            is_answerable=True,
            category="generation",
        ),
        GoldenQAPair(
            id="qa_011",
            question="What is the mathematical formula for Reciprocal Rank Fusion (RRF) and what default constant k is used in the guide?",
            expected_answer="The formula accumulates scores as 1 / (k + rank + 1) for each retrieved document across dense and sparse ranking lists, using a default constant k = 60.",
            relevant_chunk_ids=["doc_evidence-backed-rag-project-guide_1a754e0a#stratA#c0004"],
            difficulty="medium",
            is_answerable=True,
            category="retrieval",
        ),
        GoldenQAPair(
            id="qa_012",
            question="According to the strict citation prompt rules, what exact response is required if the context lacks enough information to answer confidently?",
            expected_answer="Insufficient evidence to answer this question based on the provided documents.",
            relevant_chunk_ids=["doc_evidence-backed-rag-project-guide_1a754e0a#stratA#c0004"],
            difficulty="easy",
            is_answerable=True,
            category="refusal",
        ),
        GoldenQAPair(
            id="qa_013",
            question="How is the refusal accuracy evaluation metric defined in the guide?",
            expected_answer="The percentage of deliberately unanswerable questions that are correctly refused by the system.",
            relevant_chunk_ids=["doc_evidence-backed-rag-project-guide_1a754e0a#stratA#c0004"],
            difficulty="easy",
            is_answerable=True,
            category="evaluation",
        ),
        GoldenQAPair(
            id="qa_014",
            question="Where in the project directory structure should centralized configuration settings like chunk sizes and model names reside?",
            expected_answer="In src/config.py.",
            relevant_chunk_ids=["doc_evidence-backed-rag-project-guide_1a754e0a#stratA#c0005"],
            difficulty="easy",
            is_answerable=True,
            category="architecture",
        ),
        GoldenQAPair(
            id="qa_015",
            question="What is the designated path in the project structure for the curated golden Q&A dataset?",
            expected_answer="data/golden_qa/qa_pairs.jsonl.",
            relevant_chunk_ids=["doc_evidence-backed-rag-project-guide_1a754e0a#stratA#c0005"],
            difficulty="easy",
            is_answerable=True,
            category="evaluation",
        ),
        GoldenQAPair(
            id="qa_016",
            question="What commit conventions are specified for version control in the project workflow?",
            expected_answer="Conventional commits using prefixes such as feat:, fix:, docs:, and eval:.",
            relevant_chunk_ids=["doc_evidence-backed-rag-project-guide_1a754e0a#stratA#c0006"],
            difficulty="easy",
            is_answerable=True,
            category="workflow",
        ),
        GoldenQAPair(
            id="qa_017",
            question="What file patterns must be excluded in .gitignore to ensure safety and avoid repository bloat?",
            expected_answer=".env, __pycache__/, *.pyc, large raw data files, and local vector DB files (chroma_db/).",
            relevant_chunk_ids=["doc_evidence-backed-rag-project-guide_1a754e0a#stratA#c0006"],
            difficulty="easy",
            is_answerable=True,
            category="workflow",
        ),
        GoldenQAPair(
            id="qa_018",
            question="What split deployment architecture is recommended to signal production thinking and provide an API endpoint for live demos?",
            expected_answer="Containerize the FastAPI backend with Docker and deploy it on Render or Railway, while deploying the Streamlit frontend separately on Hugging Face Spaces pointing to the backend API URL.",
            relevant_chunk_ids=["doc_evidence-backed-rag-project-guide_1a754e0a#stratA#c0007"],
            difficulty="medium",
            is_answerable=True,
            category="deployment",
        ),
        GoldenQAPair(
            id="qa_019",
            question="In what chronological order does the project brief recommend building the system components?",
            expected_answer="Ingestion -> chunking -> embedding/indexing -> hybrid retrieval -> generation with citations -> golden QA set -> evaluation harness -> API -> frontend -> Docker.",
            relevant_chunk_ids=["doc_evidence-backed-rag-project-guide_1a754e0a#stratA#c0008"],
            difficulty="medium",
            is_answerable=True,
            category="workflow",
        ),
        GoldenQAPair(
            id="qa_020",
            question="What is the primary architectural difference between bi-encoders and cross-encoders for ranking?",
            expected_answer="Bi-encoders embed queries and documents independently into vector representations, whereas cross-encoders take the query and document together to compute full joint cross-attention across all token pairs simultaneously.",
            relevant_chunk_ids=["doc_evidence-backed-rag-project-guide_1a754e0a#stratA#c0009"],
            difficulty="multi_hop",
            is_answerable=True,
            category="reranking",
        ),
        GoldenQAPair(
            id="qa_021",
            question="Why was migrating to Qdrant categorized under skipped tasks rather than Tier S?",
            expected_answer="Swapping FAISS/Chroma for Qdrant requires standing up and maintaining dedicated infrastructure for no functional gain at this scale, making it better suited as a future work item.",
            relevant_chunk_ids=["doc_evidence-backed-rag-project-guide_1a754e0a#stratA#c0010"],
            difficulty="medium",
            is_answerable=True,
            category="architecture",
        ),
        GoldenQAPair(
            id="qa_022",
            question="Which components constitute the revised Tier S must-finish deliverables?",
            expected_answer="Hybrid retrieval, reranker, golden QA set, retriever-variant comparison table (BM25/dense/hybrid/hybrid+reranker), evidence-based refusal, FastAPI, Docker, and HF Docker Space deployment.",
            relevant_chunk_ids=["doc_evidence-backed-rag-project-guide_1a754e0a#stratA#c0010"],
            difficulty="medium",
            is_answerable=True,
            category="architecture",
        ),
        GoldenQAPair(
            id="qa_023",
            question="What is the recommended number of primary shards and replica shards for an Elasticsearch cluster running this RAG pipeline?",
            expected_answer="Insufficient evidence to answer this question based on the provided documents.",
            relevant_chunk_ids=[],
            difficulty="unanswerable",
            is_answerable=False,
            category="unanswerable",
        ),
        GoldenQAPair(
            id="qa_024",
            question="How many maximum open database connections should be configured in the PostgreSQL connection pool for high concurrency?",
            expected_answer="Insufficient evidence to answer this question based on the provided documents.",
            relevant_chunk_ids=[],
            difficulty="unanswerable",
            is_answerable=False,
            category="unanswerable",
        ),
        GoldenQAPair(
            id="qa_025",
            question="What is the recommended time-to-live (TTL) in seconds for caching user session tokens in Redis?",
            expected_answer="Insufficient evidence to answer this question based on the provided documents.",
            relevant_chunk_ids=[],
            difficulty="unanswerable",
            is_answerable=False,
            category="unanswerable",
        ),
    ]

    save_golden_qa_set(pairs, target_path)
    return pairs


if __name__ == "__main__":
    dataset = create_default_golden_dataset()
    summary = validate_golden_qa_set(dataset)
    print(f"Generated {len(dataset)} golden QA pairs.")
    for key, val in summary.items():
        print(f"  {key}: {val}")
