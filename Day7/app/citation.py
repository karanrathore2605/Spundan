from typing import Any


def format_sources(retrieved_chunks: list[dict[str, Any]]) -> str:
    """
    Format sources strictly using chunk metadata.
    Does NOT allow LLM to generate source names.
    """
    if not retrieved_chunks:
        return "- No relevant sources found."

    source_lines = []
    seen_sources = set()

    for idx, chunk in enumerate(retrieved_chunks, start=1):
        source = chunk.get("source", "unknown")
        chunk_id = chunk.get("chunk_id", idx)
        identifier = f"{source} - chunk {chunk_id}"

        # Prevent duplicate identical source-chunk lines
        if identifier in seen_sources:
            continue
        seen_sources.add(identifier)

        source_lines.append(f"[{len(seen_sources)}] {identifier}")

    return "\n".join(source_lines)


def format_final_response(answer: str, retrieved_chunks: list[dict[str, Any]]) -> str:
    """
    Combine answer and source attribution into the clean Day 7 format.
    """
    sources_text = format_sources(retrieved_chunks)

    separator = "-" * 60

    return (
        f"\n{separator}\n"
        f"ANSWER\n"
        f"{separator}\n\n"
        f"{answer.strip()}\n\n"
        f"{separator}\n"
        f"SOURCES\n"
        f"{separator}\n\n"
        f"{sources_text}\n"
    )
