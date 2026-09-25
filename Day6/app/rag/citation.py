def format_sources(
    retrieved_chunks: list[dict]
) -> str:

    if not retrieved_chunks:
        return "Sources:\n- No sources available."

    lines = ["Sources:"]

    for index, chunk in enumerate(
        retrieved_chunks,
        start=1
    ):

        content = (
            chunk["content"]
            .replace("\n", " ")
            .strip()
        )

        # Keep snippets short for readability.
        max_length = 180

        if len(content) > max_length:
            content = content[:max_length].rstrip() + "..."

        lines.append(
            f"""
[{index}] {chunk["source"]}
    Chunk ID: {chunk["chunk_id"]}
    Similarity: {chunk["score"]:.4f}
    Snippet: "{content}"
"""
        )

    return "\n".join(lines)


def build_final_response(
    answer: str,
    retrieved_chunks: list[dict]
) -> str:

    sources = format_sources(
        retrieved_chunks
    )

    return f"""
Answer:
{answer}

{sources}
""".strip()