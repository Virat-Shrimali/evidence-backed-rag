"""Streamlit interactive demo interface for Evidence-Backed RAG."""

import streamlit as st

st.set_page_config(
    page_title="Evidence-Backed RAG",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 Evidence-Backed RAG System")
st.caption("Verifiable citations · Automatic refusal on insufficient evidence · Multi-strategy retrieval")

with st.sidebar:
    st.header("⚙️ Configuration")
    retriever_mode = st.selectbox(
        "Retrieval Strategy",
        options=["hybrid_rerank", "hybrid", "dense_only", "bm25_only"],
        index=0,
        help="Select which retrieval variant to use for answering.",
    )
    st.markdown("---")
    st.markdown(
        "**Guarantees:**\n"
        "- Every factual claim cites exact chunks `[chunk_id]`.\n"
        "- Questions lacking evidence strictly refuse to prevent hallucinations."
    )

st.subheader("Ask a Question")
question = st.text_input(
    "Enter your query:",
    placeholder="e.g. What are the key risk factors outlined in the documentation?",
)

if st.button("Submit Query", type="primary") and question:
    with st.spinner("Retrieving evidence and generating answer..."):
        # Scaffolding demo placeholder
        st.info("System initialized. Ingestion & index pipelines will be populated in next milestone.")
