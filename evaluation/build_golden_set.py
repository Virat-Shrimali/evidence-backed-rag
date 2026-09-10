"""Helper to validate and inspect golden QA pairs."""

from pathlib import Path

from pydantic import BaseModel, Field


class GoldenQAPair(BaseModel):
    """Schema for a golden evaluation pair."""
    id: str
    question: str
    expected_answer: str
    relevant_chunk_ids: list[str] = Field(default_factory=list)
    is_answerable: bool = True
    difficulty: str = "easy"  # easy, multi-hop, unanswerable


def load_golden_qa_set(file_path: Path) -> list[GoldenQAPair]:
    """Load and validate golden QA pairs from a JSONL file."""
    if not file_path.exists():
        return []

    pairs = []
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(GoldenQAPair.model_validate_json(line))
    return pairs
