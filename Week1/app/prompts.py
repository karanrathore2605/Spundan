from typing import Any

SYSTEM_PROMPT = """You are a RAG assistant.

Answer the user's question ONLY using the provided context.

Rules:
- Do not invent information.
- Do not use outside knowledge.
- If the answer is not available in the context, clearly say:
  "I don't have enough information in the provided documents."
- Keep the answer relevant and concise.
- Do not create or invent source names.
"""


def build_rag_prompt(question: str, retrieved_chunks: list[dict[str, Any]]) -> str:
    """
    Format retrieved document chunks and user question into a structured prompt.
    Separates context clearly from the question.
    """
    context_blocks = []

    for idx, chunk in enumerate(retrieved_chunks, start=1):
        source = chunk.get("source", "unknown")
        chunk_id = chunk.get("chunk_id", idx)
        content = chunk.get("content", "").strip()

        context_blocks.append(
            f"[Chunk {idx}] (Source: {source}, Chunk ID: {chunk_id})\n{content}"
        )

    context_str = "\n\n".join(context_blocks)

    return f"""Retrieved Context:
============================================================
{context_str}
============================================================

User Question: {question}

Grounded Answer:"""
