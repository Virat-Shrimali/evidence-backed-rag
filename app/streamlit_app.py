"""Streamlit production-ready demo UI for Evidence-Backed RAG.

Operates as a lightweight presentation layer connecting over HTTPS to the FastAPI backend.
Zero neural models, PyTorch, or vector databases are loaded in this frontend layer.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import streamlit as st

# Ensure repository root is on sys.path when executed directly by Streamlit
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Robust import supporting both package-level and direct script execution
try:
    from app.api_client import BackendClientError, get_backend_url, query_backend
except ModuleNotFoundError:
    from api_client import (  # type: ignore[no-redef]
        BackendClientError,
        get_backend_url,
        query_backend,
    )

STRATEGY_OPTIONS: dict[str, str] = {
    "Hybrid + Cross-Encoder (Tier S)": "hybrid_rerank",
    "Hybrid (Dense + BM25 RRF)": "hybrid",
    "Dense-only (all-MiniLM-L6-v2)": "dense_only",
    "BM25-only (Okapi BM25)": "bm25_only",
}

SAMPLE_QUESTIONS: list[str] = [
    "What is the core differentiator of the Evidence-Backed RAG system?",
    "What embedding model does the project use?",
    "What chunk size and overlap parameters are suggested for baseline chunking?",
    "How does PostgreSQL connection pooling work?",  # Unanswerable refusal test
]


def parse_chunk_provenance(chunk_id: str) -> dict[str, str]:
    """Parse document ID, strategy, and chunk index from a deterministic chunk ID."""
    parts = chunk_id.split("#")
    doc_id = parts[0] if len(parts) > 0 else chunk_id

    # Clean display name from doc_id (e.g. doc_evidence-backed-rag-project-guide_1a754e0a)
    doc_name = re.sub(r"^doc_", "", doc_id)
    doc_name = re.sub(r"_[a-f0-9]{8}$", "", doc_name)

    strategy_code = parts[1] if len(parts) > 1 else "unknown"
    strategy_label = (
        "Strategy A (Fixed 500)" if strategy_code == "stratA"
        else "Strategy B (Sentence 200)" if strategy_code == "stratB"
        else strategy_code
    )

    chunk_idx = parts[2] if len(parts) > 2 else ""
    return {
        "document_id": doc_id,
        "document_name": doc_name,
        "strategy": strategy_label,
        "chunk_index": chunk_idx,
    }


def format_strategy_name(strategy_key: str) -> str:
    """Format strategy internal key into user-facing label."""
    for label, key in STRATEGY_OPTIONS.items():
        if key == strategy_key:
            return label
    return strategy_key


def get_selectable_strategies(backend_url: str) -> dict[str, str]:
    """Return strategy options permitted for the target backend.

    When targeting the free Render backend (or when forced via environment variable),
    restricts selection to BM25-only to prevent triggering OOM kills on the 512 MB backend.
    Higher-memory backends (>= 2 GB RAM) expose the full multi-stage strategy suite.
    """
    is_render_free = (
        "onrender.com" in backend_url
        or os.environ.get("RENDER_FREE_MODE", "").lower() in ("true", "1", "yes")
        or os.environ.get("LOW_MEMORY_MODE", "").lower() in ("true", "1", "yes")
    )
    if is_render_free:
        return {"BM25-only (Okapi BM25) [Render Free Safe]": "bm25_only"}
    return STRATEGY_OPTIONS


def main() -> None:
    st.set_page_config(
        page_title="Evidence-Backed RAG",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    backend_url = get_backend_url()

    # Header
    st.title("📚 Evidence-Backed RAG")
    st.markdown(
        "**Ask questions over your documents. Every answer is backed by verifiable document evidence.**"
    )
    st.caption(
        "Strict chunk citations · Deterministic refusal on unanswerable queries · Multi-stage retrieval"
    )
    st.markdown("---")

    # Sidebar Controls
    with st.sidebar:
        st.header("⚙️ Retrieval & Engine Settings")

        available_strategies = get_selectable_strategies(backend_url)
        strategy_label = st.selectbox(
            "Retrieval Strategy",
            options=list(available_strategies.keys()),
            index=0,
            help="Select the retrieval and ranking architecture.",
        )
        selected_strategy = available_strategies[strategy_label]

        if len(available_strategies) == 1 and selected_strategy == "bm25_only":
            st.info(
                "🔒 **Render Free (512 MB RAM):** Operating in `bm25_only` mode to maintain low memory "
                "footprint. Dense vector search and neural Cross-Encoder reranking are preserved in the "
                "codebase for higher-memory environments (>= 2 GB RAM)."
            )

        top_k = st.slider(
            "Top Evidence Chunks (k)",
            min_value=1,
            max_value=20,
            value=5,
            help="Number of top-ranked context chunks supplied to the grounding generator.",
        )

        st.markdown("---")
        st.subheader("System Architecture")
        st.markdown(f"**Backend API:** `{backend_url}`")
        st.markdown(f"**Active Strategy:** `{selected_strategy}`")

        st.markdown("---")
        if st.button("Clear Conversation", use_container_width=True):
            st.session_state.pop("last_result", None)
            st.session_state.pop("user_question", None)
            st.rerun()

    # Query Input Section
    col_input, col_btn = st.columns([5, 1])

    with col_input:
        default_q = st.session_state.get("user_question", "")
        question = st.text_input(
            "Enter your question:",
            value=default_q,
            placeholder="e.g. What embedding model does the project use?",
            key="query_input",
        )

    with col_btn:
        st.write("")  # vertical alignment
        st.write("")
        submit = st.button("Ask Question", type="primary", use_container_width=True)

    # Sample Queries Helper
    st.caption("Quick test prompts:")
    sample_cols = st.columns(len(SAMPLE_QUESTIONS))
    for idx, sample_text in enumerate(SAMPLE_QUESTIONS):
        with sample_cols[idx]:
            is_refusal = idx == 3
            btn_label = f"🚨 {sample_text[:24]}..." if is_refusal else f"💡 {sample_text[:24]}..."
            if st.button(btn_label, key=f"sample_{idx}", help=sample_text):
                question = sample_text
                st.session_state["user_question"] = sample_text
                submit = True

    # Execution
    if submit:
        trimmed_question = question.strip()
        if not trimmed_question:
            st.warning("Please enter a non-empty question to query the document corpus.")
            return

        try:
            with st.spinner(f"Querying backend ({strategy_label}) and verifying evidence..."):
                response_data = query_backend(
                    question=trimmed_question,
                    retriever_mode=selected_strategy,
                    top_k=top_k,
                    backend_url=backend_url,
                )
                st.session_state["last_result"] = {
                    "question": trimmed_question,
                    "response": response_data,
                    "strategy": selected_strategy,
                }
        except BackendClientError as exc:
            st.error(exc.message)
            return
        except Exception:
            st.error(
                "An unexpected error occurred while communicating with the backend service. "
                "Please check the service status and try again."
            )
            return

    # Display Results
    if "last_result" in st.session_state:
        result_data = st.session_state["last_result"]
        resp = result_data["response"]
        active_strat = result_data["strategy"]
        retrieved_chunk_ids = resp.get("retrieved_chunk_ids", [])
        citations = resp.get("citations", [])

        st.markdown("---")

        # 1. Evidence Status & Confidence Bar
        col_status, col_conf = st.columns([1, 1])
        with col_status:
            if resp.get("sufficient_evidence"):
                st.success("✔ **Evidence Status:** Sufficient Evidence Grounded in Corpus")
            else:
                st.warning("⚠ **Evidence Status:** Insufficient Evidence (Deterministic Refusal)")

        with col_conf:
            conf_pct = int(resp.get("confidence", 0.0) * 100)
            st.metric(
                label="Grounding Confidence (Heuristic)",
                value=f"{conf_pct}%",
                help="Heuristic grounding confidence based on similarity thresholds; not a calibrated statistical probability.",
            )

        # 2. Answer Section
        st.subheader("Answer")
        if resp.get("sufficient_evidence"):
            st.markdown(resp.get("answer", ""))
        else:
            st.info(f"**{resp.get('answer', '')}**")
            st.caption(
                "The system could not find sufficient supporting evidence in the provided documents "
                "to ground an answer without risk of hallucination."
            )
            if resp.get("refusal_reason"):
                st.caption(f"Refusal Rationale: `{resp['refusal_reason']}`")

        # 3. Verifiable Citations Section
        if citations:
            st.subheader(f"Verifiable Citations ({len(citations)})")
            for idx, cit in enumerate(citations, 1):
                cid = cit.get("chunk_id", "")
                snippet = cit.get("text_snippet", "")
                prov = parse_chunk_provenance(cid)

                with st.container(border=True):
                    col_cid, col_doc, col_idx = st.columns([3, 2, 1])
                    with col_cid:
                        st.markdown(f"**Source / Chunk [{idx}]:** `{cid}`")
                    with col_doc:
                        st.markdown(f"**Document:** `{prov['document_name']}`")
                    with col_idx:
                        st.markdown(f"**Index:** `{prov['chunk_index'] or 'c0000'}`")

                    if snippet:
                        st.markdown(
                            f"> *\"{snippet}\"*",
                            help="Verbatim quotation verified to occur in the cited chunk text.",
                        )

        # 4. Expandable Retrieval Details & Provenance
        with st.expander("🔍 Retrieval Details & Candidate Provenance"):
            st.markdown(f"**Active Strategy:** `{format_strategy_name(active_strat)}`")
            st.markdown(f"**Total Candidates Retrieved:** `{len(retrieved_chunk_ids)}`")

            if retrieved_chunk_ids:
                for rank, cid in enumerate(retrieved_chunk_ids, 1):
                    prov = parse_chunk_provenance(cid)
                    with st.container(border=True):
                        st.markdown(
                            f"**Rank {rank} · Chunk:** `{cid}` | "
                            f"**Document:** `{prov['document_name']}` | "
                            f"**Strategy:** `{prov['strategy']}`"
                        )


if __name__ == "__main__":
    main()
