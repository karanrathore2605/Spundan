import argparse
import sys
from pathlib import Path

# Fix Windows console UTF-8 encoding issues
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from app.chunker import TextChunker
from app.config import (
    DOCUMENTS_DIR,
    MAX_QUERY_LENGTH,
    STORAGE_DIR,
    TOP_K,
    validate_groq_api_key,
)
from app.embeddings import EmbeddingService
from app.llm import GroqService
from app.loader import DocumentLoader
from app.rag_pipeline import RAGPipeline
from app.retriever import Retriever
from app.vector_store import FAISSVectorStore


def print_banner(title: str, width: int = 60) -> None:
    print("=" * width)
    print(title.center(width))
    print("=" * width)


def initialize_rag() -> RAGPipeline | None:
    """Initialize and load the complete Day 7 RAG pipeline."""
    print_banner("DAY 7 - RAG SYSTEM")
    print("\nLoading RAG pipeline...\n")

    # Edge Case 6: Validate GROQ_API_KEY
    try:
        validate_groq_api_key()
    except ValueError as exc:
        print(f"{exc}\n")
        return None

    # Load loader and check documents
    loader = DocumentLoader(DOCUMENTS_DIR)
    docs = loader.load_documents()

    # Edge Case 1: No documents found
    if not docs:
        return None

    # Initialize components
    chunker = TextChunker()
    embedding_service = EmbeddingService()
    vector_store = FAISSVectorStore()

    # Edge Case 8: Check if valid index exists on disk or rebuild
    loaded_cache = vector_store.load(STORAGE_DIR)
    if loaded_cache and vector_store.is_ready:
        chunks = vector_store.chunks
    else:
        # Build index from documents
        chunks = chunker.split(docs)
        if not chunks:
            print("Warning: No text chunks could be extracted from documents.")
            return None

        texts = [c.content for c in chunks]
        embeddings = embedding_service.generate_embeddings(texts)
        vector_store.build(embeddings, chunks)
        try:
            vector_store.save(STORAGE_DIR)
        except Exception:
            pass

    # Initialize Retriever & Groq LLM
    retriever = Retriever(embedding_service, vector_store)
    try:
        llm_service = GroqService()
    except ValueError as exc:
        print(f"{exc}\n")
        return None

    pipeline = RAGPipeline(
        loader=loader,
        chunker=chunker,
        embedding_service=embedding_service,
        vector_store=vector_store,
        retriever=retriever,
        llm_service=llm_service,
        storage_dir=STORAGE_DIR,
    )

    print(f"Documents loaded: {len(docs)}")
    print(f"Chunks created: {len(chunks)}")
    print("Vector index ready.")

    return pipeline


def interactive_loop(pipeline: RAGPipeline) -> None:
    """Run interactive question-answering CLI."""
    print("\n" + "=" * 60)
    print("RAG SYSTEM READY".center(60))
    print("=" * 60)
    print("\nAsk a question.")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            user_input = input("Question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        # Edge Case 10: User exits
        if user_input.lower() in {"exit", "quit", "q"}:
            print("\nGoodbye!")
            break

        # Edge Case 3: Empty question
        if not user_input:
            print("Please enter a valid question.\n")
            continue

        # Guard against excessively long questions
        if len(user_input) > MAX_QUERY_LENGTH:
            print(f"Question is too long (max {MAX_QUERY_LENGTH} characters). Please shorten it.\n")
            continue

        print("\nRetrieving relevant information...")
        print("Generating grounded answer...")

        response = pipeline.ask(user_input, top_k=TOP_K)
        print(response)


def main() -> int:
    parser = argparse.ArgumentParser(description="Day 7 Unified RAG System")
    parser.add_argument(
        "--question",
        "-q",
        type=str,
        help="Optional single question to run in non-interactive mode",
    )
    args = parser.parse_args()

    pipeline = initialize_rag()
    if pipeline is None:
        return 1

    if args.question:
        print("\nRetrieving relevant information...")
        print("Generating grounded answer...")
        response = pipeline.ask(args.question, top_k=TOP_K)
        print(response)
        return 0

    interactive_loop(pipeline)
    return 0


if __name__ == "__main__":
    sys.exit(main())
