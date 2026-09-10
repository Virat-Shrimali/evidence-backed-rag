# Evidence-Backed RAG System + Evaluation Harness

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

> A production-grade Question Answering system where **every claim cites exact retrieved chunks**, and the system strictly returns **"Insufficient evidence to answer this question based on the provided documents."** when grounding is inadequate — eliminating ungrounded hallucinations. Accompanied by a rigorous **empirical evaluation harness** benchmarking dense, sparse (BM25), hybrid (RRF), and cross-encoder reranking strategies.

---

## 1. Problem & Motivation

Standard RAG pipelines often suffer from two major production failure modes:
1. **Plausible Hallucinations:** Generating convincing yet ungrounded answers when retrieved documents contain only tangentially related text.
2. **Opaque Provenance:** Providing answers without verifiable, chunk-level provenance that human reviewers or domain experts can immediately audit.

This system solves both problems by enforcing structured chunk-level citations (`[chunk_id]`), implementing an automated evidence sufficiency check with refusal logic, and measuring retrieval and generation quality against a hand-crafted golden benchmark.

---

## 2. Technical Architecture

```
                     ┌─────────────────────┐
                     │   Document Corpus   │
                     │ (PDFs / Text Docs)  │
                     └──────────┬──────────┘
                                │
                        1. Parse & Clean
                                │
                        2. Chunk (Strategy A / B)
                                │
                    ┌───────────┴────────────┐
                    │                        │
            3a. Dense Embeddings      3b. Sparse Index
            (all-MiniLM-L6-v2)           (Rank-BM25)
                    │                        │
                    └───────────┬────────────┘
                                │
                      4. Hybrid Retriever
                    (Reciprocal Rank Fusion)
                                │
                      5. Cross-Encoder Reranker
                 (ms-marco-MiniLM-L-6-v2: Top-20 → Top-5)
                                │
                      6. Context-Enriched Prompt
                   (Strict citation & grounding rules)
                                │
                      7. Evidence Sufficiency Check
                      ┌─────────┴─────────┐
                      │                   │
                  [Sufficient]      [Insufficient]
                      │                   │
               Return Answer +     Return Exact Standard
               Chunk Citations     Refusal Message
```

---

## 3. Evaluation & Experimental Results

The evaluation harness evaluates multiple retrieval strategies against a golden Q&A dataset with single-chunk, multi-chunk, and intentionally unanswerable questions:

| Retrieval Strategy | Recall@5 | Precision@5 | MRR | Faithfulness | Answer Correctness | Refusal Accuracy | p95 Latency |
|---|---|---|---|---|---|---|---|
| **BM25 Only** | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |
| **Dense Only (MiniLM)** | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |
| **Hybrid (Dense + BM25 RRF)** | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |
| **Hybrid + Reranker (Tier S)** | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |

*Detailed benchmark metrics, breakdown by question difficulty, and comparative analysis are generated dynamically in [evaluation/results/comparison_table.md](evaluation/results/comparison_table.md).*

---

## 4. Repository Structure

```
evidence-backed-rag/
├── README.md                      # Case-study style system documentation
├── requirements.txt               # Pinned production and development dependencies
├── pyproject.toml                 # Packaging, build system, test and lint configuration
├── .gitignore                     # Git exclusions (caches, vector DBs, secrets)
├── .env.example                   # Environment variable template
├── Dockerfile                     # Containerization for deployment
│
├── data/
│   ├── raw/                       # Source document corpus
│   ├── processed/                 # Parsed and cleaned text files
│   └── golden_qa/
│       └── qa_pairs.jsonl         # Hand-crafted golden QA evaluation benchmark
│
├── src/
│   ├── __init__.py
│   ├── config.py                  # Centralized system configurations and thresholds
│   ├── ingest/
│   │   ├── __init__.py
│   │   ├── parse.py               # Document parsing and text normalization
│   │   └── chunk.py               # Swappable chunking strategies (Fixed vs Sentence)
│   ├── index/
│   │   ├── __init__.py
│   │   ├── embed.py               # Dense vector indexing with ChromaDB
│   │   └── bm25_index.py          # Sparse keyword indexing
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── hybrid.py              # Reciprocal Rank Fusion (RRF) retriever
│   │   └── rerank.py              # Cross-Encoder reranker
│   ├── generation/
│   │   ├── __init__.py
│   │   ├── prompt_templates.py    # Strict citation-enforcing prompt templates
│   │   └── generate.py            # Structured generation and refusal logic
│   └── pipeline.py                # End-to-end query execution pipeline
│
├── evaluation/
│   ├── __init__.py
│   ├── retrieval_metrics.py       # Precision@K, Recall@K, MRR metrics
│   ├── ragas_eval.py              # Ragas faithfulness, relevancy, and correctness
│   ├── run_comparison.py          # Multi-strategy benchmark runner
│   └── results/
│       └── comparison_table.md    # Generated empirical comparison report
│
├── api/
│   ├── __init__.py
│   └── main.py                    # FastAPI server exposing POST /query
│
├── app/
│   ├── __init__.py
│   └── streamlit_app.py           # Interactive Streamlit UI with citation inspector
│
└── tests/
    ├── __init__.py
    ├── conftest.py                # Pytest fixtures and test doubles
    ├── test_foundation.py         # Sanity and configuration tests
    ├── test_chunking.py           # Chunking strategy unit tests
    ├── test_retrieval.py          # Dense, sparse, and RRF unit tests
    └── test_generation.py         # Citation schema and refusal unit tests
```

---

## 5. Quickstart

### Prerequisites
- Python 3.11+
- Git

### Setup
```bash
# Clone the repository
git clone <repo-url>
cd evidence-backed-rag

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
```

### Running Tests
```bash
pytest
```

---

## 6. License
MIT License.
