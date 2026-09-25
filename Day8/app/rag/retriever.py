"""
Retriever module orchestrating the end-to-end RAG retrieval pipeline.
Provides structured document search over indexed vector content.
"""

from pathlib import Path
from typing import Optional
import logging

from app.rag.loader import DocumentLoader
from app.rag.chunker import TextChunker
from app.rag.embeddings import EmbeddingEngine
from app.rag.vector_store import FAISSVectorStore
from app.schemas.tool_schema import SearchResponse, SearchResultChunk

logger = logging.getLogger(__name__)


class RAGRetriever:
    """
    Coordinates document loading, chunking, embedding generation,
    and FAISS similarity search.
    """

    def __init__(
        self,
        documents_dir: Path,
        embedding_engine: Optional[EmbeddingEngine] = None,
        chunk_size: int = 300,
        chunk_overlap: int = 50,
        top_k: int = 3,
    ):
        self.documents_dir = Path(documents_dir)
        self.embedding_engine = embedding_engine or EmbeddingEngine()
        self.chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.top_k = top_k
        self._vector_store: Optional[FAISSVectorStore] = None
        self._is_indexed = False

    @property
    def vector_store(self) -> FAISSVectorStore:
        """Initializes and returns the vector store."""
        if self._vector_store is None:
            self._vector_store = FAISSVectorStore(dimension=self.embedding_engine.dimension)
        return self._vector_store

    def build_index(self, force_reload: bool = False) -> int:
        """
        Loads documents, chunks them, computes embeddings, and populates FAISS.
        
        Args:
            force_reload: Whether to clear existing index and reload.
            
        Returns:
            int: Number of chunks indexed.
        """
        if self._is_indexed and not force_reload:
            return self.vector_store.count()

        logger.info("Building RAG retrieval index from %s ...", self.documents_dir)
        loader = DocumentLoader(self.documents_dir)
        documents = loader.load()

        if not documents:
            logger.warning("No documents found in %s to index.", self.documents_dir)
            self._is_indexed = True
            return 0

        chunks = self.chunker.chunk_documents(documents)
        if not chunks:
            logger.warning("No chunks generated from documents.")
            self._is_indexed = True
            return 0

        texts = [c.content for c in chunks]
        embeddings = self.embedding_engine.encode_batch(texts)

        if force_reload and self._vector_store is not None:
            self._vector_store.clear()

        self.vector_store.add_chunks(chunks, embeddings)
        self._is_indexed = True
        logger.info("Successfully indexed %d chunks across %d documents.", len(chunks), len(documents))
        return len(chunks)

    def retrieve(self, query: str, top_k: Optional[int] = None) -> SearchResponse:
        """
        Executes dense vector search for a natural language query.
        
        Args:
            query: Natural language query string.
            top_k: Optional override for number of chunks to retrieve.
            
        Returns:
            SearchResponse: Structured results with content, source, and score.
        """
        if not query or not query.strip():
            raise ValueError("Query string cannot be empty or whitespace.")

        clean_query = query.strip()
        k = top_k or self.top_k

        # Ensure index is built
        if not self._is_indexed or self.vector_store.count() == 0:
            self.build_index()

        if self.vector_store.count() == 0:
            logger.warning("Vector store is empty. Returning 0 results.")
            return SearchResponse(query=clean_query, results=[], total_found=0)

        query_emb = self.embedding_engine.encode_text(clean_query)
        raw_matches = self.vector_store.search(query_emb, top_k=k)

        results = [
            SearchResultChunk(
                content=chunk.content,
                source=chunk.source,
                score=round(score, 4),
            )
            for chunk, score in raw_matches
        ]

        return SearchResponse(
            query=clean_query,
            results=results,
            total_found=len(results),
        )
