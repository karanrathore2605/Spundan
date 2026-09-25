"""
Vector store module using FAISS for dense vector similarity indexing.
"""

from typing import List, Tuple
import faiss
import numpy as np
import logging
from app.rag.chunker import TextChunk

logger = logging.getLogger(__name__)


class FAISSVectorStore:
    """In-memory FAISS vector index storing text chunks and embeddings."""

    def __init__(self, dimension: int):
        self.dimension = dimension
        # IndexFlatIP computes exact inner products (cosine similarity for normalized vectors)
        self.index = faiss.IndexFlatIP(dimension)
        self.chunks: List[TextChunk] = []

    def count(self) -> int:
        """Returns the number of indexed vectors."""
        return len(self.chunks)

    def add_chunks(self, chunks: List[TextChunk], embeddings: np.ndarray) -> None:
        """
        Adds text chunks and their corresponding embeddings to the FAISS index.
        
        Args:
            chunks: List of TextChunk objects.
            embeddings: 2D numpy array of shape (N, dimension), float32.
        """
        if len(chunks) == 0:
            logger.warning("No chunks to add to vector store.")
            return

        if len(chunks) != embeddings.shape[0]:
            raise ValueError(
                f"Chunk count ({len(chunks)}) does not match embedding count ({embeddings.shape[0]})"
            )

        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension ({embeddings.shape[1]}) does not match index dimension ({self.dimension})"
            )

        # FAISS requires float32 contiguous array
        float32_embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        self.index.add(float32_embeddings)
        self.chunks.extend(chunks)
        logger.info("Added %d chunks to FAISS index. Total index size: %d", len(chunks), self.count())

    def search(self, query_embedding: np.ndarray, top_k: int = 3) -> List[Tuple[TextChunk, float]]:
        """
        Performs nearest-neighbor search for a query embedding.
        
        Args:
            query_embedding: 2D array of shape (1, dimension).
            top_k: Number of nearest neighbors to retrieve.
            
        Returns:
            List of tuples: (TextChunk, similarity_score).
        """
        if self.count() == 0:
            logger.warning("Search attempted on empty vector store.")
            return []

        k = min(top_k, self.count())
        float32_query = np.ascontiguousarray(query_embedding, dtype=np.float32)
        scores, indices = self.index.search(float32_query, k)

        results: List[Tuple[TextChunk, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self.chunks):
                results.append((self.chunks[idx], float(score)))

        return results

    def clear(self) -> None:
        """Resets the index and stored chunks."""
        self.index.reset()
        self.chunks.clear()
        logger.info("FAISS vector store cleared.")
