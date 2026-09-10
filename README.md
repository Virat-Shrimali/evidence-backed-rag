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
    ├── test_deployment.py         # Container & deployment configuration tests
    └── test_ui.py                 # Streamlit demo UI helper unit tests
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

## 6. Streamlit Interactive Demo UI

An interactive, recruiter-friendly presentation UI is provided in `app/streamlit_app.py` to visually demonstrate the core guarantees of the Evidence-Backed RAG system.

### Launching the UI
```bash
streamlit run app/streamlit_app.py
```

### Key Capabilities Demonstrated
1. **Verifiable Chunk Citations:** Every factual claim cites exact `[chunk_id]` identifiers with document name, page numbers, and verbatim highlighted quotations.
2. **Deterministic Refusal:** Asking unanswerable questions demonstrates immediate, deterministic refusal without hallucination.
3. **Multi-Strategy Comparison:** Interactively switch between **BM25-only**, **Dense-only**, **Hybrid (RRF)**, and **Hybrid + Cross-Encoder** to observe real-time ranking and latency differences.
4. **Candidate Provenance Inspection:** Expandable retrieval details allow full audit of candidate chunks, similarity scores, and neural reranking outputs.

### Architecture & Relationship to FastAPI
Streamlit acts strictly as a presentation/demo layer. Both the Streamlit UI and the FastAPI REST service share the identical underlying `RAGPipeline` (`src/pipeline.py`). Business logic, retrieval fusion, reranking, and citation validation are not duplicated across services.

---

## 7. Containerization & Deployment: Phase 1 (FastAPI Backend Service)

The project follows a **two-phase deployment strategy**:
- **Phase 1 (Active):** Containerize and deploy the production **FastAPI REST API** as a standalone backend on a Hugging Face Docker Space.
- **Phase 2:** Deploy the interactive Streamlit presentation UI connected to the live backend service.

### Container Architecture & Optimizations
- **Base Image:** `python:3.11-slim` (minimal attack surface and lightweight image).
- **Security & User:** Non-root execution with UID `1000` (`useradd -m -u 1000 user`), ensuring full compliance with Hugging Face Spaces security sandbox.
- **Cache Pre-configuration:** Dedicated `/home/user/.cache` directory owned by `user:user` with `HF_HOME=/home/user/.cache/huggingface` and `TORCH_HOME=/home/user/.cache/torch` to prevent permission errors when downloading models.
- **Lean Runtime Dependencies:** Builds using `requirements-backend.txt`, excluding development, evaluation harness (`ragas`), and UI (`streamlit`) dependencies to keep image size minimal and build times fast.
- **Dynamic Port & Host Binding:** Defaults to `PORT=7860` (standard for Hugging Face Spaces) and binds to `0.0.0.0`, adapting dynamically to `${PORT}`.
- **Auto-Indexing on Cold Container Start:** On container startup, if the Chroma vector store or BM25 index is unpopulated (due to persistence directories being ignored in `.dockerignore`), the pipeline automatically parses and indexes the documents in `data/raw/` on first query.

### Hugging Face Free Tier Resource Profiling
| Resource | Space Allocation | Pipeline Consumption | Status |
|---|---|---|---|
| **vCPU** | 2 vCPU | ~2.8s CPU inference for cross-encoder reranking | Well within capacity |
| **RAM** | 16 GB | ~1.2 GB peak (MiniLM dense + ms-marco cross-encoder) | ~7.5% utilization |
| **Disk** | 50 GB | ~1.5 GB (Python 3.11-slim + PyTorch CPU + models) | ~3% utilization |
| **GPU** | Optional (T4 available) | CPU default; works completely without GPU | CPU fully verified |

### API Endpoints
- `GET /health`: Instant readiness check returning `{"status": "healthy", "service": "evidence-backed-rag"}`.
- `POST /query`: Grounded Q&A endpoint accepting `{"question": "..."}` and returning a structured `RAGResponse` with verifiable citations.
- `GET /docs`: Interactive Swagger OpenAPI documentation.
- `GET /redoc`: Alternative ReDoc API documentation.

### Sample API Requests & Responses

#### 1. Health Readiness Check
```bash
curl -X GET "https://<hf-username>-<space-name>.hf.space/health"
```
```json
{
  "status": "healthy",
  "service": "evidence-backed-rag",
  "pipeline_initialized": true
}
```

#### 2. Answerable Document-Grounded Query
```bash
curl -X POST "https://<hf-username>-<space-name>.hf.space/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the core differentiator of the Evidence-Backed RAG system?"}'
```

#### 3. Deterministic Refusal on Unanswerable Query
```bash
curl -X POST "https://<hf-username>-<space-name>.hf.space/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the capital of Atlantis?"}'
```
Response:
```json
{
  "answer": "Insufficient evidence to answer this question based on the provided documents.",
  "citations": [],
  "evidence_found": false,
  "confidence": 0.0
}
```

### Environment Configuration & Secrets
> [!IMPORTANT]
> Never hardcode or commit secrets into images or git repositories. All sensitive credentials must be supplied via runtime environment variables or Space Secrets.

| Environment Variable | Default | Description |
|---|---|---|
| `PORT` | `7860` | Server listening port (injected automatically by Hugging Face) |
| `LLM_PROVIDER` | `mock` | Generation backend (`mock`, `openai`, `anthropic`, `ollama`) |
| `OPENAI_API_KEY` | `""` | Required when `LLM_PROVIDER=openai` |
| `ANTHROPIC_API_KEY` | `""` | Required when `LLM_PROVIDER=anthropic` |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Endpoint when `LLM_PROVIDER=ollama` |
| `LLM_MODEL_NAME` | `gpt-4o-mini` | Target LLM model identifier |
| `RETRIEVAL_STRATEGY`| `hybrid_rerank` | Active retrieval strategy (`dense_only`, `bm25_only`, `hybrid`, `hybrid_rerank`) |

### Hugging Face Space Deployment Steps (Phase 1)
1. Create a new Space on [Hugging Face](https://huggingface.co/new-space).
2. Set Space Name (e.g. `evidence-backed-rag-api`) and choose **Docker** as the Space SDK (Blank template).
3. Connect your Git repository or push directly to the Space:
   ```bash
   git remote add hf-backend https://huggingface.co/spaces/<your-username>/evidence-backed-rag-api
   git push hf-backend feat/backend-deployment:main
   ```
4. In Space **Settings → Variables and secrets**:
   - Add Secret `OPENAI_API_KEY` (if using OpenAI).
   - Add Variable `LLM_PROVIDER=openai` (or leave default `mock` for zero-cost testing).
5. Hugging Face automatically detects `sdk: docker` and `app_port: 7860` from the README YAML frontmatter, builds the container with `Dockerfile`, and starts the Uvicorn server.
6. The public backend URL will be accessible at:
   `https://<your-username>-evidence-backed-rag-api.hf.space`

> [!NOTE]
> **Host Environment Status:** The automated offline test suite (87 tests), end-to-end smoke test, and Ruff static linting are 100% verified. In the local development environment, the Docker Desktop daemon was not running due to local Windows host permissions (`com.docker.service` stopped); container specifications, non-root user setup, environment variable mapping, health check handlers, and lean runtime dependencies are verified via static checks and automated unit tests.

---

## 8. System Limitations & Production Considerations

1. **Benchmark Scale:** The current golden QA dataset is an initial high-quality development benchmark (26 hand-crafted pairs). For enterprise deployments, this should be expanded to hundreds of representative domain questions with automated continuous evaluation in CI.
2. **Inference Latency:** Neural cross-encoder reranking (`ms-marco-MiniLM-L-6-v2`) runs on CPU by default, requiring ~2.8s per query over 20 candidates. In high-throughput production environments, latency can be reduced to under 50ms using GPU inference, ONNX Runtime, or quantization (e.g., INT8/FP16).
3. **Document Formats:** Ingestion currently supports digital PDF and clean text documents. Scanned paper documents containing raster images require an additional OCR extraction step (e.g., Tesseract or cloud document AI).
4. **Offline vs Live LLM Generation:** Offline automated testing and smoke verification use deterministic mock providers (`MockLLMProvider`) to ensure zero-cost, 100% reproducible tests. Production deployments require active provider credentials (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or local `ollama`).

---

## 9. License
MIT License.


