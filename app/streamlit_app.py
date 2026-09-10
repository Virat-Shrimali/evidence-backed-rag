"""Streamlit production-ready demo UI for Evidence-Backed RAG."""

from __future__ import annotations

import re

import streamlit as st

from src.config import settings
from src.pipeline import RAGPipeline

STRATEGY_OPTIONS: dict[str, str] = {
    "Hybrid + Cross-Encoder (Tier S)": "hybrid_rerank",
    "Hybrid (Dense + BM25 RRF)": "hybrid",
    "Dense-only (all-MiniLM-L6-v2)": "dense_only",
    "BM25-only (Okapi BM25)": "bm25_only",
}

SAMPLE_QUESTIONS: list[str] = [
    "What is the core differentiator of the Evidence-Backed RAG system?",
    "Which embedding models are recommended for local vector generation?",
    "What chunk size and overlap parameters are suggested for baseline chunking?",
    "What is the secret recipe for Martian hot chocolate?",  # Unanswerable refusal test
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


@st.cache_resource(show_spinner="Initializing Evidence-Backed RAG Pipeline...")
def get_pipeline() -> RAGPipeline:
    """Lazily initialize and cache the unified RAGPipeline instance."""
    return RAGPipeline()


def main() -> None:
    st.set_page_config(
        page_title="Evidence-Backed RAG",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded",
    )

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

        strategy_label = st.selectbox(
            "Retrieval Strategy",
            options=list(STRATEGY_OPTIONS.keys()),
            index=0,
            help="Select the retrieval and ranking architecture.",
        )
        selected_strategy = STRATEGY_OPTIONS[strategy_label]

        top_k = st.slider(
            "Top Evidence Chunks (k)",
            min_value=1,
            max_value=20,
            value=5,
            help="Number of top-ranked context chunks supplied to the grounding generator.",
        )

        st.markdown("---")
        st.subheader("System Architecture")
        st.markdown(f"**LLM Backend:** `{settings.llm_provider}`")
        st.markdown(f"**Model:** `{settings.llm_model_name}`")
        st.markdown(f"**Confidence Threshold:** `{settings.evidence_confidence_threshold}`")

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
            placeholder="e.g. What is the core differentiator of the Evidence-Backed RAG system?",
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
            btn_label = f"🚨 {sample_text[:28]}..." if is_refusal else f"💡 {sample_text[:28]}..."
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

        pipeline = get_pipeline()
        try:
            with st.spinner(f"Executing retrieval ({strategy_label}) and verifying evidence..."):
                response, candidates = pipeline.query_with_candidates(
                    question=trimmed_question,
                    retriever_mode=selected_strategy,
                    top_k=top_k,
                )
                st.session_state["last_result"] = {
                    "question": trimmed_question,
                    "response": response,
                    "candidates": candidates,
                    "strategy": selected_strategy,
                }
        except Exception as exc:
            st.error(f"An error occurred while executing the query: {exc}")
            return

    # Display Results
    if "last_result" in st.session_state:
        result_data = st.session_state["last_result"]
        resp = result_data["response"]
        candidates = result_data["candidates"]
        active_strat = result_data["strategy"]

        st.markdown("---")

        # 1. Evidence Status & Confidence Bar
        col_status, col_conf = st.columns([1, 1])
        with col_status:
            if resp.sufficient_evidence:
                st.success("✔ **Evidence Status:** Sufficient Evidence Grounded in Corpus")
            else:
                st.warning("⚠ **Evidence Status:** Insufficient Evidence (Deterministic Refusal)")

        with col_conf:
            conf_pct = int(resp.confidence * 100)
            st.metric(
                label="Grounding Confidence (Heuristic)",
                value=f"{conf_pct}%",
                help="Heuristic grounding confidence based on similarity thresholds; not a calibrated statistical probability.",
            )

        # 2. Answer Section
        st.subheader("Answer")
        if resp.sufficient_evidence:
            st.markdown(resp.answer)
        else:
            st.info(f"**{resp.answer}**")
            st.caption(
                "The system could not find sufficient supporting evidence in the provided documents to ground an answer without risk of hallucination."
            )
            if resp.refusal_reason:
                st.caption(f"Refusal Rationale: `{resp.refusal_reason}`")

        # 3. Verifiable Citations Section
        if resp.citations:
            st.subheader(f"Verifiable Citations ({len(resp.citations)})")
            candidate_map = {c.chunk_id: c for c in candidates}

            for idx, cit in enumerate(resp.citations, 1):
                prov = parse_chunk_provenance(cit.chunk_id)
                cand = candidate_map.get(cit.chunk_id)
                pages_str = (
                    f"Page {cand.page_numbers}" if cand and cand.page_numbers
                    else "Page 1"
                )

                with st.container(border=True):
                    col_cid, col_doc, col_pg = st.columns([3, 2, 1])
                    with col_cid:
                        st.markdown(f"**Citation [{idx}]:** `{cit.chunk_id}`")
                    with col_doc:
                        st.markdown(f"**Document:** `{prov['document_name']}`")
                    with col_pg:
                        st.markdown(f"**Location:** {pages_str}")

                    if cit.text_snippet:
                        st.markdown(
                            f"> *\"{cit.text_snippet}\"*",
                            help="Verbatim quotation verified to occur in the cited chunk text.",
                        )

        # 4. Expandable Retrieval Details & Provenance
        with st.expander("🔍 Retrieval Details & Candidate Provenance"):
            st.markdown(f"**Active Strategy:** `{format_strategy_name(active_strat)}`")
            st.markdown(f"**Total Candidates Retrieved:** `{len(candidates)}`")

            if candidates:
                for rank, chunk in enumerate(candidates, 1):
                    with st.container(border=True):
                        st.markdown(
                            f"**Rank {rank} · Chunk:** `{chunk.chunk_id}` | "
                            f"**Initial Score:** `{chunk.score:.4f}`"
                            + (f" | **Rerank Score:** `{chunk.rerank_score:.4f}`" if chunk.rerank_score is not None else "")
                        )
                        st.text(chunk.content[:300] + ("..." if len(chunk.content) > 300 else ""))


if __name__ == "__main__":
    main()
