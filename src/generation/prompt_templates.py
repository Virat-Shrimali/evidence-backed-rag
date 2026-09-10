"""Prompt templates enforcing strict chunk citations and refusal on weak evidence."""

STRICT_RAG_SYSTEM_PROMPT = """You are answering questions using ONLY the provided context chunks.

Rules:
1. Every claim in your answer must be traceable to a specific chunk.
2. Cite chunks inline using [chunk_id].
3. If the context does not contain enough information to answer confidently, respond exactly with:
   "Insufficient evidence to answer this question based on the provided documents."
4. Do not use outside knowledge, even if you know the answer.

Respond in this JSON format:
{
  "answer": "...",
  "citations": [{"chunk_id": "...", "snippet": "..."}],
  "sufficient_evidence": true/false,
  "confidence": 0.0 to 1.0,
  "refusal_reason": null or string
}"""


def format_context_prompt(chunks: list, question: str) -> str:
    """Format retrieved chunks and question into context prompt."""
    formatted_chunks = []
    for chunk in chunks:
        formatted_chunks.append(f"--- Chunk ID: {chunk.id} ---\n{chunk.content}")
    context_text = "\n\n".join(formatted_chunks)

    return f"""Context:
{context_text}

Question: {question}

Remember to respond in the required JSON format only."""
