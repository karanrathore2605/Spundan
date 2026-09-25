from src.chunking.text_splitter import TextSplitter
from src.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DOCUMENTS_DIR,
    MIN_CHUNK_SIZE,
    TOP_K,
)
from src.embeddings.embedding_service import EmbeddingService
from src.llm.gemini_service import GeminiService
from src.loaders.document_loader import DocumentLoader
from src.rag.rag_pipeline import RAGPipeline
from src.retrieval.retriever import Retriever
from src.vectorstore.faiss_store import FAISSStore


def build_rag_pipeline() -> RAGPipeline:
    """
    Build the complete RAG pipeline.
    """

    print("\nLoading documents...")

    # -----------------------------------------------------
    # 1. Load documents
    # -----------------------------------------------------

    loader = DocumentLoader(
        DOCUMENTS_DIR
    )

    documents = loader.load_documents()

    print(
        f"Loaded {len(documents)} document sections."
    )

    # -----------------------------------------------------
    # 2. Split documents into chunks
    # -----------------------------------------------------

    print("Creating chunks...")

    splitter = TextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        min_chunk_size=MIN_CHUNK_SIZE,
    )

    chunks = splitter.split_documents(
        documents
    )

    print(
        f"Created {len(chunks)} chunks."
    )

    if not chunks:
        raise RuntimeError(
            "No usable chunks were created."
        )

    # -----------------------------------------------------
    # 3. Generate embeddings
    # -----------------------------------------------------

    print("Generating embeddings...")

    embedding_service = EmbeddingService()

    texts = [
        chunk.text
        for chunk in chunks
    ]

    embeddings = (
        embedding_service.generate_embeddings(
            texts
        )
    )

    print(
        f"Generated embeddings with shape: "
        f"{embeddings.shape}"
    )

    # -----------------------------------------------------
    # 4. Build FAISS vector store
    # -----------------------------------------------------

    print("Building FAISS index...")

    vector_store = FAISSStore()

    vector_store.build(
        embeddings=embeddings,
        chunks=chunks,
    )

    # -----------------------------------------------------
    # 5. Create Retriever
    # -----------------------------------------------------

    retriever = Retriever(
        embedding_service=embedding_service,
        vector_store=vector_store,
        top_k=TOP_K,
    )

    # -----------------------------------------------------
    # 6. Create Gemini service
    # -----------------------------------------------------

    print("Connecting to Gemini...")

    gemini_service = GeminiService()

    # -----------------------------------------------------
    # 7. Create RAG pipeline
    # -----------------------------------------------------

    pipeline = RAGPipeline(
        retriever=retriever,
        llm=gemini_service,
    )

    return pipeline


def print_response(
    answer: str,
    sources,
) -> None:

    print("\n")
    print("=" * 70)
    print("FINAL ANSWER")
    print("=" * 70)

    print(answer)

    print("\n")
    print("=" * 70)
    print("SOURCES")
    print("=" * 70)

    for index, result in enumerate(
        sources,
        start=1,
    ):

        chunk = result.chunk

        page = (
            f"Page {chunk.page}"
            if chunk.page is not None
            else "Page N/A"
        )

        print(
            f"{index}. "
            f"{chunk.source} | "
            f"{page} | "
            f"Score: {result.score:.4f}"
        )

    print("=" * 70)


def main():

    print("=" * 70)
    print("                 DAY 5 - MINI RAG")
    print("=" * 70)

    try:

        pipeline = build_rag_pipeline()

        print("\nRAG system is ready.")
        print("Type 'exit' to quit.")

        while True:

            question = input(
                "\nAsk a question:\n> "
            ).strip()

            if question.lower() == "exit":
                print("\nGoodbye!")
                break

            if not question:
                print(
                    "Please enter a question."
                )
                continue

            try:

                print(
                    "\nSearching documents..."
                )

                response = pipeline.ask(
                    question
                )

                print_response(
                    response.answer,
                    response.sources,
                )

            except Exception as exc:

                print(
                    f"\nError while processing "
                    f"question: {exc}"
                )

    except Exception as exc:

        print(
            f"\nApplication startup failed: {exc}"
        )


if __name__ == "__main__":
    main()