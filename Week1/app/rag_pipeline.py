import logging
from pathlib import Path
from typing import Any

from app.chunker import TextChunker
from app.citation import format_final_response
from app.config import (
    DOCUMENTS_DIR,
    STORAGE_DIR,
    TOP_K,
    sanitize_top_k,
)
from app.embeddings import EmbeddingService
from app.llm import GroqService
from app.loader import DocumentLoader
from app.models import Document
from app.prompts import SYSTEM_PROMPT, build_rag_prompt
from app.retriever import Retriever
from app.vector_store import FAISSVectorStore

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Unified end-to-end RAG Pipeline integrating:
    Document Loading -> Chunking -> Embedding -> FAISS Storage -> Retrieval -> Groq LLM -> Citation.
    """

    def __init__(
        self,
        loader: DocumentLoader,
        chunker: TextChunker,
        embedding_service: EmbeddingService,
        vector_store: FAISSVectorStore,
        retriever: Retriever,
        llm_service: GroqService,
        storage_dir: Path | str = STORAGE_DIR,
    ):
        self.loader = loader
        self.chunker = chunker
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.retriever = retriever
        self.llm_service = llm_service
        self.storage_dir = Path(storage_dir)

    @classmethod
    def create(
        cls,
        documents_dir: Path | str = DOCUMENTS_DIR,
        storage_dir: Path | str = STORAGE_DIR,
        llm_service: GroqService | None = None,
    ) -> "RAGPipeline":
        """Factory method to assemble a complete pipeline with default components."""
        loader = DocumentLoader(documents_dir)
        chunker = TextChunker()
        embedding_service = EmbeddingService()
        vector_store = FAISSVectorStore()
        retriever = Retriever(embedding_service, vector_store)
        
        # If llm_service is not supplied, create default GroqService
        if llm_service is None:
            llm_service = GroqService()

        return cls(
            loader=loader,
            chunker=chunker,
            embedding_service=embedding_service,
            vector_store=vector_store,
            retriever=retriever,
            llm_service=llm_service,
            storage_dir=storage_dir,
        )

    def ingest_documents(self, documents: list[Document]) -> int:
        """Process and embed a list of Document objects into the FAISS vector store."""
        if not documents:
            return 0

        chunks = self.chunker.split(documents)
        if not chunks:
            return 0

        texts = [chunk.content for chunk in chunks]
        embeddings = self.embedding_service.generate_embeddings(texts)
        self.vector_store.build(embeddings, chunks)

        return len(chunks)

    def ingest_document(self, file_path: Path | str) -> int:
        """
        Load a single text file, chunk it, embed it, and add it to the vector store.
        """
        doc = self.loader.load_document(file_path)
        if not doc:
            return 0

        chunks = self.chunker.split_document(doc)
        if not chunks:
            return 0

        texts = [chunk.content for chunk in chunks]
        embeddings = self.embedding_service.generate_embeddings(texts)
        self.vector_store.add(embeddings, chunks)
        return len(chunks)

    def ingest_directory(self, documents_dir: Path | str | None = None) -> tuple[int, int]:
        """
        Load all documents from directory, chunk them, embed, and initialize vector store.
        Returns (document_count, chunk_count).
        """
        if documents_dir is not None:
            self.loader.documents_dir = Path(documents_dir)

        docs = self.loader.load_documents()
        if not docs:
            return 0, 0

        chunk_count = self.ingest_documents(docs)
        return len(docs), chunk_count

    def ask(self, question: str, top_k: int = TOP_K) -> str:
        """
        Answer a user question end-to-end:
        1. Validates question
        2. Retrieves relevant chunks from FAISS
        3. Invokes Groq LLM with grounded prompt
        4. Returns grounded answer + sources
        """
        # Edge Case 3: Empty question
        if not question or not question.strip():
            return "Please enter a valid question."

        clean_question = question.strip()
        safe_top_k = sanitize_top_k(top_k)

        # Edge Case 8: Ensure vector store is ready, rebuild if empty/corrupted
        if not self.vector_store.is_ready:
            logger.info("Vector store not ready. Attempting to build from documents...")
            docs_count, chunk_count = self.ingest_directory()
            # Edge Case 1: Still no documents found
            if docs_count == 0 or chunk_count == 0:
                return (
                    "No documents found.\n"
                    "Please add a text document first."
                )

        # Retrieve relevant chunks
        retrieved_chunks = self.retriever.retrieve(clean_question, top_k=safe_top_k)

        # Edge Case 5: No relevant chunks found above similarity threshold
        if not retrieved_chunks:
            return format_final_response(
                "No relevant information was found in the documents.",
                [],
            )

        # Build prompt and query LLM
        prompt = build_rag_prompt(clean_question, retrieved_chunks)
        llm_response = self.llm_service.generate(
            prompt=prompt,
            system_instruction=SYSTEM_PROMPT,
        )

        # Handle empty LLM generation or failure
        if not llm_response:
            llm_response = "Unable to generate the answer because the LLM service is currently unavailable."

        # Return formatted Answer + Sources (Edge Case 4 and general flow)
        return format_final_response(llm_response, retrieved_chunks)
