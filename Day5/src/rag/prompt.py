from src.models import RetrievedChunk


SYSTEM_INSTRUCTIONS = """
You are a document question-answering assistant.

Your job is to answer the user's question using ONLY the
provided document context.

Rules:
1. Do not use outside knowledge.
2. Do not invent facts.
3. If the answer is not available in the context, say:
   "I could not find this information in the provided documents."
4. Give a clear and concise answer.
5. Use bullet points when useful.
6. When possible, refer to the source using [Source N].
"""


def build_rag_prompt(
    question: str,
    retrieved_chunks: list[RetrievedChunk],
) -> str:

    context_parts = []

    for index, result in enumerate(
        retrieved_chunks,
        start=1,
    ):

        chunk = result.chunk

        page_info = (
            f"Page {chunk.page}"
            if chunk.page is not None
            else "Page information unavailable"
        )

        context_parts.append(
            f"""
[Source {index}]
Document: {chunk.source}
{page_info}
Similarity Score: {result.score:.4f}

Content:
{chunk.text}
"""
        )

    context = "\n".join(context_parts)

    prompt = f"""
{SYSTEM_INSTRUCTIONS}

DOCUMENT CONTEXT
================

{context}

USER QUESTION
=============

{question}

ANSWER
======

Answer the question using only the document context above.
"""

    return prompt