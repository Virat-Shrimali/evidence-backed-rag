# Evidence-Backed RAG System + Evaluation Harness
## Full Technical Build Guide

---

## 0. What you're building (one paragraph)

A Q&A system over a real document corpus where **every answer cites the exact chunk it came from**, and the system explicitly says **"insufficient evidence to answer"** when it can't ground a claim — instead of hallucinating. Paired with an **evaluation harness**: a golden set of 50–100 Q&A pairs used to measure retrieval quality and answer correctness across at least 2 different chunking/embedding/retrieval strategies, so you can show *experimental* evidence for your design choices, not just a working demo.

This one paragraph should also become your resume bullet once you have numbers to plug in.

---

## 1. Tech Stack

| Layer | Choice | Why |
|---|---|---|
| Language | Python 3.11+ | ecosystem standard |
| Orchestration | Plain Python first, LangChain optional later | build retrieval by hand first so you understand it — bolt on LangChain only if you want the abstraction on your resume too |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2` or `bge-small-en-v1.5`), free, local, no API cost | avoids API bills during heavy experimentation |
| Vector store | `ChromaDB` (local, zero-infra) or `FAISS` (lighter, no server) | both are free and resume-recognized; Chroma is easier to start with |
| Keyword search (for hybrid) | `rank_bm25` | classic sparse retrieval to combine with dense vectors |
| LLM (generation) | Anthropic Claude API or OpenAI API (pay-as-you-go, cheap at this scale) OR local via `Ollama` (Llama 3.1 8B / Mistral) if you want $0 cost | pick API for quality, Ollama for a "no external dependency" version — mention both as a design tradeoff in your README |
| Reranker (stretch goal) | `cross-encoder/ms-marco-MiniLM-L-6-v2` via `sentence-transformers` | cheap way to add "reranking" to your resume bullet |
| PDF/doc parsing | `pypdf` or `unstructured` | depends on corpus format |
| Backend/API | `FastAPI` | industry standard, async, auto docs |
| Frontend/demo | `Streamlit` | fastest way to a clickable demo |
| Evaluation | `ragas` library (context precision/recall, faithfulness, answer correctness) + your own script for retrieval P/R@K | ragas is the current standard RAG-eval library and recruiters recognize the name |
| Experiment tracking | Simple CSV/JSON logs + a `results/` folder with markdown tables, OR `Weights & Biases` free tier if you want it on resume | don't over-engineer this part |
| Env/dependency mgmt | `venv` + `requirements.txt`, or `uv` (faster, modern) | keep it simple and reproducible |
| Version control | Git + GitHub | see Section 6 |
| Containerization | `Docker` (optional but strong signal) | for deployment consistency |
| Deployment | Hugging Face Spaces (Streamlit/Gradio, free, easiest) or Render (FastAPI backend, free tier) | see Section 7 |

---

## 2. Architecture

```
                     ┌─────────────────────┐
                     │   Document Corpus    │
                     │ (PDFs / policies /   │
                     │  10-Ks / syllabus)   │
                     └──────────┬───────────┘
                                │
                        1. Parse & Clean
                                │
                        2. Chunk (strategy A/B)
                                │
                    ┌───────────┴────────────┐
                    │                         │
            3a. Embed chunks          3b. Build BM25 index
            (dense vectors)             (sparse index)
                    │                         │
                    └───────────┬─────────────┘
                                │
                      4. Hybrid Retriever
                   (dense + sparse, merged/weighted)
                                │
                    5. (optional) Reranker
                                │
                       6. Top-K chunks
                                │
              7. LLM Prompt (question + chunks + citation rules)
                                │
                    8. Answer + Citations
                                │
              9. Confidence check: enough evidence?
                     │                    │
                  YES → return         NO → "insufficient
                  answer + sources       evidence to answer"
```

**Evaluation harness runs in parallel:**

```
Golden Q&A set (50–100 pairs, hand-written by you)
        │
        ├── Run through Strategy A (e.g. chunk_size=500, MiniLM embeddings)
        ├── Run through Strategy B (e.g. chunk_size=200, bge-small embeddings)
        │
        ▼
Compute per strategy:
  - Retrieval: Precision@K, Recall@K
  - Generation: faithfulness, answer correctness (via ragas)
  - Ops: latency, token cost
        │
        ▼
results/comparison_table.md  ← this table IS your differentiator
```

---

## 3. Step-by-step build plan (realistic timeline: ~2–3 weeks part-time)

**Phase 1 — Corpus & baseline (2–3 days)**
1. Pick ONE corpus with real substance (suggestions: your university's course syllabi + past papers, 5–10 public company 10-K filings, WHO/CDC medical guidelines, a set of research papers on one topic). 30–100 documents is plenty.
2. Parse to clean text. Handle tables/headers if present.
3. Chunk with a simple fixed strategy first (e.g. 500 tokens, 50 overlap) just to get an end-to-end pipeline working.
4. Embed with `all-MiniLM-L6-v2`, store in Chroma.
5. Build a bare BM25 index over the same chunks.
6. Write a basic hybrid retriever: pull top-K from both, merge with reciprocal rank fusion (simple and well-documented).
7. Send retrieved chunks + question to the LLM with a strict prompt (Section 4 has the prompt template).
8. Get a working CLI or notebook demo end-to-end. Don't polish yet.

**Phase 2 — Citations & the "insufficient evidence" logic (2–3 days)**
9. Force the LLM to output structured JSON: `{answer, citations: [{chunk_id, text_snippet}], confidence}`.
10. Add a rule: if retrieved chunks' similarity scores are all below a threshold, OR the LLM itself flags low confidence, return "insufficient evidence to answer" instead of a guess.
11. Test this deliberately — ask it something NOT in your corpus and confirm it refuses correctly. This is your best demo moment in interviews.

**Phase 3 — Evaluation harness (4–5 days, this is the differentiator)**
12. Hand-write 50–100 Q&A pairs from your corpus. Mix easy (single-chunk) and hard (multi-chunk/synthesis) questions. Include a few "should be unanswerable" questions.
13. Implement retrieval metrics: Precision@K, Recall@K (you know the "correct" chunk(s) for each golden question, so this is straightforward set comparison).
14. Wire up `ragas` for faithfulness and answer correctness against your golden answers.
15. Define Strategy B (different chunk size and/or different embedding model and/or dense-only vs hybrid).
16. Run both strategies through the full eval set, log latency and (if using a paid API) token cost.
17. Produce a comparison table and a short written conclusion ("Strategy B improved Recall@5 by 12% at the cost of 40ms extra latency, because...").

**Phase 4 — Polish, deploy, document (3–4 days)**
18. Build the Streamlit front end (chat-style UI showing answer + expandable citations).
19. Wrap retrieval+generation in a FastAPI backend if you want a proper API layer.
20. Dockerize (optional but a nice resume line).
21. Deploy (Section 7).
22. Write the README as a case study (Section 8 has the template).

---

## 4. Key implementation details

### 4.1 Chunking strategies to compare (pick 2)
- Fixed-size (e.g. 500 tokens / 50 overlap) — baseline
- Semantic/recursive splitting (split on headers/paragraphs, `RecursiveCharacterTextSplitter` if using LangChain, or roll your own)
- Smaller chunks (e.g. 200 tokens) to see precision/recall tradeoff

### 4.2 Hybrid retrieval — reciprocal rank fusion (simple, effective, easy to explain in interviews)
```python
def reciprocal_rank_fusion(dense_results, sparse_results, k=60):
    scores = {}
    for rank, doc_id in enumerate(dense_results):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
    for rank, doc_id in enumerate(sparse_results):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: -x[1])
```

### 4.3 Strict citation prompt template
```
You are answering questions using ONLY the provided context chunks.

Rules:
1. Every claim in your answer must be traceable to a specific chunk.
2. Cite chunks inline using [chunk_id].
3. If the context does not contain enough information to answer
   confidently, respond exactly with:
   "Insufficient evidence to answer this question based on the
   provided documents."
4. Do not use outside knowledge, even if you know the answer.

Context:
{retrieved_chunks_with_ids}

Question: {question}

Respond in this JSON format:
{{
  "answer": "...",
  "citations": [{{"chunk_id": "...", "snippet": "..."}}],
  "sufficient_evidence": true/false
}}
```

### 4.4 Evaluation metrics to report
- **Retrieval:** Precision@K, Recall@K, MRR
- **Generation (via ragas):** faithfulness, answer_relevancy, answer_correctness
- **Refusal accuracy:** % of deliberately unanswerable questions correctly refused (this metric alone is a great talking point)
- **Ops:** p50/p95 latency, cost per query (if using paid API)

---

## 5. File structure

```
evidence-backed-rag/
├── README.md                      ← case-study style writeup (Section 8)
├── requirements.txt
├── .gitignore
├── .env.example                   ← API key placeholders, never commit real keys
├── Dockerfile
├── docker-compose.yml             ← optional, if backend+frontend run together
│
├── data/
│   ├── raw/                       ← original PDFs/docs (gitignored if large)
│   ├── processed/                 ← cleaned text
│   └── golden_qa/
│       └── qa_pairs.jsonl         ← your 50–100 hand-written Q&A pairs
│
├── src/
│   ├── __init__.py
│   ├── config.py                  ← chunk sizes, model names, thresholds — centralized
│   ├── ingest/
│   │   ├── __init__.py
│   │   ├── parse.py               ← PDF/doc → clean text
│   │   └── chunk.py               ← chunking strategies A and B
│   ├── index/
│   │   ├── __init__.py
│   │   ├── embed.py               ← embedding + Chroma indexing
│   │   └── bm25_index.py          ← sparse index
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── hybrid.py              ← reciprocal rank fusion
│   │   └── rerank.py              ← optional cross-encoder reranker
│   ├── generation/
│   │   ├── __init__.py
│   │   ├── prompt_templates.py
│   │   └── generate.py            ← LLM call + JSON parsing + citation logic
│   └── pipeline.py                ← end-to-end orchestration: query → answer
│
├── evaluation/
│   ├── __init__.py
│   ├── build_golden_set.py        ← helper to generate/validate QA pairs
│   ├── retrieval_metrics.py       ← Precision@K, Recall@K, MRR
│   ├── ragas_eval.py              ← faithfulness, answer correctness
│   ├── run_comparison.py          ← runs Strategy A vs B end-to-end, logs results
│   └── results/
│       ├── strategy_a_results.json
│       ├── strategy_b_results.json
│       └── comparison_table.md    ← generated summary, link this from README
│
├── api/
│   └── main.py                    ← FastAPI app exposing /query endpoint
│
├── app/
│   └── streamlit_app.py           ← demo UI
│
├── notebooks/
│   └── exploration.ipynb          ← scratch work, not the source of truth
│
└── tests/
    ├── test_chunking.py
    ├── test_retrieval.py
    └── test_generation.py
```

---

## 6. Version control & best practices

- **Repo setup:** one GitHub repo, clear name (`evidence-backed-rag`, not `project1`).
- **Branching:** `main` always working. Feature branches: `feat/hybrid-retrieval`, `feat/eval-harness`, `feat/streamlit-ui`. Merge via PRs even solo — it looks deliberate and you can write good commit history.
- **Commit style:** conventional commits — `feat:`, `fix:`, `docs:`, `eval:` (e.g. `eval: add strategy B comparison results`). Recruiters who click into your repo notice this.
- **`.gitignore`:** must exclude `.env`, `__pycache__/`, `*.pyc`, large raw data files, local vector DB files (`chroma_db/` or similar) unless small.
- **Secrets:** never commit API keys. Use `.env` + `python-dotenv`, provide `.env.example` with placeholder names only.
- **README-driven:** write the README skeleton on day 1, fill it in as you go — this forces you to think about the story, not just the code.
- **Tests:** even 5–10 basic unit tests (chunking produces expected chunk count, retrieval returns K results, JSON output parses correctly) shows engineering discipline disproportionate to the effort.
- **requirements.txt:** pin versions (`sentence-transformers==3.0.1` not just `sentence-transformers`) for reproducibility.

---

## 7. Deployment

**Easiest path (recommended for campus placement timelines):**
1. Push code to GitHub.
2. Deploy the Streamlit app directly on **Hugging Face Spaces** (free, supports Streamlit natively, just needs `requirements.txt` + `app.py`). If your vector index is small, bundle a pre-built Chroma DB in the Space; otherwise rebuild it on Space startup.
3. If using a paid LLM API, add your key as a Space "secret" (not in code).

**More impressive path (if time allows):**
1. Containerize with Docker (`Dockerfile` builds both FastAPI backend and installs deps).
2. Deploy FastAPI backend on **Render** (free tier) or **Railway**.
3. Deploy Streamlit frontend separately on HF Spaces, pointing to your Render API URL.
4. This split (frontend/backend) is a stronger signal of "production thinking" and gives you a real API endpoint to show in interviews (`curl` demo).

**Either way:** put the live demo link at the top of your README and on your resume next to this project.

---

## 8. README template (write this as a case study, not documentation)

```markdown
# Evidence-Backed RAG System

[Live Demo](link) | [Video walkthrough (optional, 2 min Loom)](link)

## Problem
[1–2 sentences: what question were you answering, why grounding/citations matter]

## Approach
[Short paragraph: corpus, chunking, hybrid retrieval, citation-forcing prompt]

## Results
| Metric | Strategy A (baseline) | Strategy B |
|---|---|---|
| Recall@5 | X% | Y% |
| Faithfulness | X | Y |
| Answer correctness | X | Y |
| Refusal accuracy (unanswerable Qs) | X% | Y% |
| p95 latency | Xms | Yms |

[2–3 sentences interpreting WHY strategy B won/lost — this is what shows real understanding]

## Architecture
[diagram or link to Section 2 above]

## Tech stack
[bullet list]

## What I'd do next
[1–2 honest limitations / next steps — shows maturity]
```

---

## 9. Project brief for an AI coding agent (Antigravity / Claude Code / Cursor etc.)

Paste this as your initial prompt to scaffold the repo — adjust the corpus description to your actual choice first.

```
Build a Python project called "evidence-backed-rag" implementing a
retrieval-augmented generation system with a paired evaluation harness.

Requirements:
- Python 3.11, use venv + requirements.txt
- Corpus: [DESCRIBE YOUR CHOSEN CORPUS HERE, e.g. "10 public company
  10-K filings in data/raw/ as PDFs"]
- Implement two chunking strategies (fixed-size 500/50 overlap, and
  smaller 200-token chunks) as swappable config in src/config.py
- Embeddings via sentence-transformers (all-MiniLM-L6-v2), stored in
  ChromaDB
- Sparse retrieval via rank_bm25 over the same chunks
- Hybrid retrieval combining both via reciprocal rank fusion
- LLM generation step that:
  - only answers from provided context
  - returns structured JSON with answer, citations (chunk_id +
    snippet), and a sufficient_evidence boolean
  - returns "Insufficient evidence to answer this question based on
    the provided documents." when evidence is weak
- An evaluation harness in evaluation/ that:
  - loads a golden QA set from data/golden_qa/qa_pairs.jsonl
    (fields: question, expected_answer, relevant_chunk_ids,
    is_answerable)
  - computes Precision@K and Recall@K per strategy
  - uses the `ragas` library to compute faithfulness and answer
    correctness
  - runs both chunking strategies end-to-end and writes a comparison
    table to evaluation/results/comparison_table.md
- FastAPI backend (api/main.py) exposing POST /query
- Streamlit frontend (app/streamlit_app.py) with a chat UI showing
  answer + expandable source citations
- Dockerfile for the FastAPI service
- Unit tests for chunking, retrieval, and generation JSON parsing
- Follow the exact file structure below: [PASTE SECTION 5 FILE TREE]
- Use conventional commits and set up the repo with a main branch and
  a feat/hybrid-retrieval branch to start

Build it in this order: ingestion → chunking → embedding/indexing →
hybrid retrieval → generation with citations → golden QA set →
evaluation harness → API → frontend → Docker. Confirm each stage works
with a quick test before moving to the next.
```

---

## 10. Turning results into resume bullets (fill in once you have numbers)

- "Built an evidence-backed RAG system over [N] documents with hybrid BM25+dense retrieval; achieved [X]% answer correctness and [Y]% recall@5 on a hand-built 100-question evaluation set (ragas, custom retrieval metrics)."
- "Designed and ran a chunking/embedding comparison (2 strategies) showing a [X]% recall improvement at [Y]ms added latency, informing a documented production tradeoff."
- "Implemented citation-grounded generation with automatic refusal on insufficient evidence, achieving [X]% refusal accuracy on deliberately unanswerable questions — reducing hallucination risk vs. baseline RAG."
- "Deployed the system as a live demo (FastAPI + Streamlit, Dockerized) on [Render/HF Spaces], with API and interactive UI."

These are the exact sentences interviewers will ask you to defend — make sure every number in them is one you can explain, not just one that sounds good.

---

## 11. Addendum — adopted upgrades & realistic priority tiers

A few refinements worth folding in, kept deliberately scoped for a 2–4 week campus timeline (coursework included).

**Adopt (cheap, high signal):**
- Add a **cross-encoder reranker** (`cross-encoder/ms-marco-MiniLM-L-6-v2`) after your hybrid retrieval step, top-20 → top-5.
- Extend your evaluation matrix to compare **BM25-only vs dense-only vs hybrid vs hybrid+reranker** (not just chunking strategy A vs B) — same harness, one more axis, and it's the single strongest table in the whole project.
- Extend the generation JSON schema with `confidence`, `retrieved_chunk_ids`, `refusal_reason`.
- Deploy via **HF Docker Space** (not the native Streamlit SDK — HF deprecated that in April 2025; pick the Docker SDK + Streamlit template).
- Add a minimal **GitHub Actions** workflow: `pytest` + `ruff` on push. ~20 minutes of YAML, disproportionate signal.

**Skip unless you have real spare time (don't let these block shipping):**
- Swapping FAISS/Chroma for **Qdrant** — real infra to stand up and keep alive for no functional gain at this scale. Fine as a "what I'd do next" line in your README instead.
- **MLflow** — a `results/` folder with JSON + a markdown comparison table tells the same story in an interview for a fraction of the setup time.
- **Docker Compose with FastAPI + Qdrant + frontend as separate services** — real networking/health-check work. One Dockerfile for FastAPI, Streamlit calling it internally, deployed as a single HF Docker Space, gets ~90% of the "production" signal for ~30% of the effort.
- Combining PyMuPDF *and* Unstructured — pick one (Unstructured alone is enough for a scoped corpus).

**Revised Tier S (must finish):** hybrid retrieval → reranker → golden QA set → retriever-variant comparison table (BM25/dense/hybrid/hybrid+reranker) → evidence-based refusal → FastAPI → Docker → HF Docker Space deployment.

**Tier A (do if on schedule):** GitHub Actions CI, richer JSON schema, latency/cost tracking, 10–20 unit tests.

**Tier B (only if genuinely ahead of schedule):** Qdrant migration, MLflow, parent-child chunking, multi-provider LLM benchmark, separate frontend/backend deployment.
