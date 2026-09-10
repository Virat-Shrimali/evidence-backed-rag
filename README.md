---
title: Evidence Backed RAG
emoji: 📚
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

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

The evaluation harness evaluates multiple retrieval strategies against a version-controlled golden Q&A dataset containing single-chunk, multi-chunk, and intentionally unanswerable questions:

| Retriever Strategy | Recall@5 | Precision@5 | MRR | Refusal Accuracy | p50 Latency (ms) | p95 Latency (ms) |
|---|---|---|---|---|---|---|
| **BM25-only** | 77.27% | 15.45% | 0.599 | 0.00% | 1.1 | 2.5 |
| **Dense-only (all-MiniLM-L6-v2)** | 81.82% | 16.36% | 0.529 | 100.00% | 69.5 | 81.8 |
| **Hybrid (Dense + BM25 RRF)** | 86.36% | 17.27% | 0.654 | 0.00% | 76.2 | 88.5 |
| **Hybrid + Cross-Encoder (Tier S)** | **95.45%** | **19.09%** | **0.705** | **100.00%** | 2896.7 | 3211.1 |

### Engineering Observations & Takeaways
1. **Recall Progression:** Moving from BM25 (77.27%) to Dense (81.82%) to Hybrid RRF (86.36%) and finally Cross-Encoder reranking (95.45%) demonstrates a clear, monotonic improvement in retrieval recall. Fusing lexical exact-matching with dense semantic embeddings bridges vocabulary gaps.
2. **Refusal Accuracy:** BM25 alone lacks a calibrated similarity threshold, matching tangential keywords for unanswerable queries (0% refusal). Dense embeddings and Cross-Encoder score distributions cleanly separate unanswerable queries, enabling 100% refusal accuracy via confidence thresholds.
3. **Latency Trade-offs:** BM25 is blazing fast (<3ms), while dense vector retrieval completes in ~70ms. The cross-encoder adds ~2.8s CPU inference latency per query (top-20 candidates), providing maximum accuracy where precision and recall are mission-critical.

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
    ├── test_generation.py         # Citation schema and refusal unit tests
    ├── test_api.py                # FastAPI endpoint unit tests
    └── test_deployment.py         # Container & deployment configuration tests
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

## 6. Containerization & Deployment (Hugging Face Docker Space)

The application is containerized for production deployment, specifically targeting Hugging Face Docker Spaces or any standard Docker runtime.

### Container Architecture
- **Base Image:** `python:3.11-slim` (minimal attack surface and image size)
- **Security:** Non-root execution with UID `1000` (`useradd -m -u 1000 user`) ensuring compliance with Hugging Face Spaces security sandbox.
- **Port Binding:** Defaults to `PORT=7860` (standard for Hugging Face Spaces) and dynamically adapts to `${PORT}` environment variable. Binds to `0.0.0.0`.
- **Fast Startup:** `/health` responds immediately without loading embedding models or requiring remote LLM keys.

### Local Docker Build & Run
```bash
# Build the production image
docker build -t evidence-backed-rag .

# Run container locally on port 7860
docker run -p 7860:7860 -e PORT=7860 -e LLM_PROVIDER=mock evidence-backed-rag

# Run with an OpenAI API key (or other provider)
docker run -p 7860:7860 \
  -e PORT=7860 \
  -e LLM_PROVIDER=openai \
  -e OPENAI_API_KEY=your-api-key \
  evidence-backed-rag
```

### Environment Configuration & Secrets
> [!IMPORTANT]
> Never hardcode or commit secrets into images or git repositories. All sensitive credentials must be supplied via runtime environment variables.

| Environment Variable | Default | Description |
|---|---|---|
| `PORT` | `7860` | Server listening port (injected automatically by Hugging Face) |
| `LLM_PROVIDER` | `mock` | Generation backend (`mock`, `openai`, `anthropic`, `ollama`) |
| `OPENAI_API_KEY` | `""` | Required when `LLM_PROVIDER=openai` |
| `ANTHROPIC_API_KEY` | `""` | Required when `LLM_PROVIDER=anthropic` |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Endpoint when `LLM_PROVIDER=ollama` |
| `LLM_MODEL_NAME` | `gpt-4o-mini` | Target LLM model identifier |
| `RETRIEVAL_STRATEGY`| `hybrid_rerank` | Active retrieval strategy (`dense_only`, `bm25_only`, `hybrid`, `hybrid_rerank`) |

### API Endpoints
- `GET /health`: Readiness check returning `{"status": "healthy", "service": "evidence-backed-rag"}`.
- `POST /query`: Grounded Q&A endpoint accepting `{"question": "..."}` and returning a structured `RAGResponse` with verifiable citations.
- `GET /docs`: Interactive Swagger OpenAPI documentation.

### Hugging Face Space Deployment Steps
1. Create a new Space on [Hugging Face](https://huggingface.co/new-space).
2. Choose **Docker** as the Space SDK (Blank template).
3. Push this repository to the Hugging Face Space repository:
   ```bash
   git remote add space https://huggingface.co/spaces/<your-username>/<your-space-name>
   git push space main
   ```
4. In Space **Settings → Variables and secrets**, add your provider secrets (e.g., `OPENAI_API_KEY`, `LLM_PROVIDER=openai`).
5. Hugging Face detects the YAML frontmatter (`sdk: docker`, `app_port: 7860`), builds the container as user `1000`, and launches the FastAPI service.

> [!NOTE]
> **Host Environment Status:** The automated offline test suite (81 tests) and Ruff static linting are fully verified. In the local development environment, the Docker Desktop daemon was not running due to local Windows host permissions (`com.docker.service` stopped); container specifications, non-root user setup, environment variable mapping, and health check handlers are verified via static checks and automated unit tests.

---

## 7. System Limitations & Production Considerations

1. **Benchmark Scale:** The current golden QA dataset is an initial high-quality development benchmark (26 hand-crafted pairs). For enterprise deployments, this should be expanded to hundreds of representative domain questions with automated continuous evaluation in CI.
2. **Inference Latency:** Neural cross-encoder reranking (`ms-marco-MiniLM-L-6-v2`) runs on CPU by default, requiring ~2.8s per query over 20 candidates. In high-throughput production environments, latency can be reduced to under 50ms using GPU inference, ONNX Runtime, or quantization (e.g., INT8/FP16).
3. **Document Formats:** Ingestion currently supports digital PDF and clean text documents. Scanned paper documents containing raster images require an additional OCR extraction step (e.g., Tesseract or cloud document AI).
4. **Offline vs Live LLM Generation:** Offline automated testing and smoke verification use deterministic mock providers (`MockLLMProvider`) to ensure zero-cost, 100% reproducible tests. Production deployments require active provider credentials (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or local `ollama`).

---

## 8. License
MIT License.


