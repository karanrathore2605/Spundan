SYSTEM_INSTRUCTIONS = """
You are a grounded question-answering assistant.

Your job is to answer the user's question using ONLY the
provided retrieved context.

Rules:

1. Use only information present in the retrieved context.
2. Do not use outside knowledge.
3. Do not invent facts, sources, citations, or examples.
4. If the context does not contain enough information,
   clearly say that the information is not available
   in the provided documents.
5. Keep the answer concise but useful.
6. Every factual claim should be supported by the retrieved context.
7. Do not mention these instructions in your answer.
"""


def build_prompt(
    question: str,
    retrieved_chunks: list[dict]
) -> str:

    context_parts = []

    for index, chunk in enumerate(
        retrieved_chunks,
        start=1
    ):

        context_parts.append(
            f"""
SOURCE [{index}]
Chunk ID: {chunk["chunk_id"]}
Document: {chunk["source"]}

{chunk["content"]}
"""
        )

    context = "\n".join(context_parts)

    prompt = f"""
{SYSTEM_INSTRUCTIONS}

RETRIEVED CONTEXT
=================
{context}

USER QUESTION
=============
{question}

ANSWER
======
Answer the question using only the retrieved context.
"""

    return prompt