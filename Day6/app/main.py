from app.chunking.text_splitter import TextChunker
from app.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DOCUMENTS_DIR,
    TOP_K,
)
from app.embeddings.embedding_service import EmbeddingService
from app.llm.gemini_service import GeminiService
from app.loaders.document_loader import DocumentLoader
from app.rag.rag_pipeline import RAGPipeline
from app.retrieval.retriever import Retriever
from app.vectorstore.faiss_store import FAISSVectorStore


def build_rag_pipeline():

    print("=" * 70)
    print("DAY 6 - GROUNDED MINI RAG")
    print("=" * 70)

    # ============================================================
    # 1. LOAD DOCUMENTS
    # ============================================================

    print("\n[1/5] Loading documents...")

    loader = DocumentLoader(DOCUMENTS_DIR)

    documents = loader.load_documents()

    if not documents:
        raise RuntimeError(
            f"No documents found in: {DOCUMENTS_DIR}"
        )

    print(
        f"Loaded {len(documents)} document(s)."
    )

    # ============================================================
    # 2. CREATE CHUNKS
    # ============================================================

    print("\n[2/5] Creating chunks...")

    chunker = TextChunker(
        chunk_size=CHUNK_SIZE,
        overlap=CHUNK_OVERLAP,
    )

    chunks = chunker.split(documents)

    if not chunks:
        raise RuntimeError(
            "No chunks were created from the documents."
        )

    print(
        f"Created {len(chunks)} chunks."
    )

    # Show a small preview for debugging
    print("\nChunk preview:")

    for chunk in chunks[:3]:

        print("-" * 70)

        print(
            f"Chunk ID: "
            f"{chunk.metadata.get('chunk_id', 'N/A')}"
        )

        print(
            f"Source: "
            f"{chunk.metadata.get('source', 'Unknown')}"
        )

        preview = chunk.content.replace(
            "\n",
            " "
        ).strip()

        if len(preview) > 200:
            preview = (
                preview[:200]
                .rsplit(" ", 1)[0]
                + "..."
            )

        print(
            f"Content: {preview}"
        )

    # ============================================================
    # 3. GENERATE EMBEDDINGS
    # ============================================================

    print("\n[3/5] Generating embeddings...")

    embedding_service = EmbeddingService()

    texts = [
        chunk.content
        for chunk in chunks
    ]

    if not texts:
        raise RuntimeError(
            "No text available for embedding."
        )

    embeddings = (
        embedding_service
        .generate_embeddings(texts)
    )

    print(
        f"Embedding shape: {embeddings.shape}"
    )

    # ============================================================
    # 4. BUILD FAISS INDEX
    # ============================================================

    print("\n[4/5] Building FAISS index...")

    vector_store = FAISSVectorStore()

    vector_store.build(
        embeddings,
        chunks,
    )

    print("FAISS index ready.")

    # ============================================================
    # 5. INITIALIZE RETRIEVER + GEMINI + RAG
    # ============================================================

    print("\n[5/5] Initializing Gemini...")

    retriever = Retriever(
        embedding_service,
        vector_store,
    )

    gemini_service = GeminiService()

    rag_pipeline = RAGPipeline(
        retriever,
        gemini_service,
    )

    print("RAG pipeline ready.")

    return rag_pipeline


def main():

    try:

        # ========================================================
        # BUILD RAG PIPELINE
        # ========================================================

        rag_pipeline = build_rag_pipeline()

        # ========================================================
        # QUESTION / ANSWER LOOP
        # ========================================================

        print("\n" + "=" * 70)
        print("ASK QUESTIONS")
        print("Type 'exit' or 'quit' to stop.")
        print("=" * 70)

        while True:

            question = input(
                "\nQuestion: "
            ).strip()

            # ----------------------------------------------------
            # Exit
            # ----------------------------------------------------

            if question.lower() in {
                "exit",
                "quit",
            }:

                print("\nGoodbye!")
                break

            # ----------------------------------------------------
            # Empty question
            # ----------------------------------------------------

            if not question:

                print(
                    "Please enter a question."
                )

                continue

            # ----------------------------------------------------
            # Ask RAG
            # ----------------------------------------------------

            try:

                response = rag_pipeline.ask(
                    question,
                    TOP_K,
                )

                print("\n" + "-" * 70)
                print("ANSWER")
                print("-" * 70)

                print(response)

                print("-" * 70)

            except Exception as exc:

                print(
                    "\nError while processing question:"
                )

                print(
                    f"{type(exc).__name__}: {exc}"
                )

                print(
                    "Please try another question."
                )

    except KeyboardInterrupt:

        print(
            "\n\nApplication stopped by user."
        )

    except Exception as exc:

        print(
            "\nApplication startup failed:"
        )

        print(
            f"{type(exc).__name__}: {exc}"
        )


if __name__ == "__main__":
    main()