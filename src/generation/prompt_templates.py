"""Prompt templates enforcing strict chunk citations and refusal on weak evidence."""

from typing import Any

STRICT_RAG_SYSTEM_PROMPT = """You are a strictly evidence-grounded Question Answering assistant.
You must answer questions using ONLY the supplied retrieved context chunks.

Mandatory Rules:
1. Answer ONLY using the supplied retrieved context chunks.
2. Do NOT use outside knowledge, assumptions, or extrapolated facts, even if you know the answer.
3. Every factual claim in your answer must be supported by retrieved evidence.
4. Cite the exact chunk_id(s) supporting each claim.
5. Include a short, verbatim text snippet for each citation demonstrating where the fact was stated.
6. If the context does not contain sufficient evidence to answer confidently, refuse rather than guess.
   When refusing, respond exactly with:
   "Insufficient evidence to answer this question based on the provided documents."

Output Format:
You MUST respond with valid JSON adhering to this exact schema:
{
  "answer": "...",
  "citations": [
    {
      "chunk_id": "exact_chunk_id",
      "text_snippet": "verbatim snippet from cited chunk"
    }
  ],
  "confidence": 0.0 to 1.0,
  "sufficient_evidence": true/false,
  "refusal_reason": null or "explanation if insufficient evidence"
}"""


def format_context_prompt(chunks: list[Any], question: str) -> str:
    """Format retrieved chunks and question into context prompt."""
    formatted_chunks: list[str] = []
    for chunk in chunks:
        cid = getattr(chunk, "chunk_id", getattr(chunk, "id", str(chunk)))
        content = getattr(chunk, "content", "")
        formatted_chunks.append(f"--- Chunk ID: {cid} ---\n{content}")

    context_text = "\n\n".join(formatted_chunks) if formatted_chunks else "No retrieved context available."

    return f"""Context:
{context_text}

Question: {question}

Remember to respond in the required JSON format only."""

