"""
Embedding module for the RAG pipeline.
Uses Sentence Transformers to generate normalized dense vector representations.
"""

from typing import List, Optional
import numpy as np
import logging
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingEngine:
    """Manages text embedding generation using SentenceTransformer."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model: Optional[SentenceTransformer] = None
        self._dimension: Optional[int] = None

    @property
    def model(self) -> SentenceTransformer:
        """Lazy load the sentence transformer model."""
        if self._model is None:
            logger.info("Loading embedding model: %s ...", self.model_name)
            self._model = SentenceTransformer(self.model_name)
            # Determine vector dimension
            dummy_vec = self._model.encode(["test"], normalize_embeddings=True)
            self._dimension = int(dummy_vec.shape[1])
            logger.info("Embedding model loaded. Dimension: %d", self._dimension)
        return self._model

    @property
    def dimension(self) -> int:
        """Vector dimensionality of the embedding model."""
        if self._dimension is None:
            _ = self.model  # Trigger lazy loading
        return self._dimension  # type: ignore

    def encode_text(self, text: str) -> np.ndarray:
        """
        Generates a normalized embedding vector for a single text string.
        
        Args:
            text: Query or chunk text.
            
        Returns:
            np.ndarray of shape (1, dimension), float32.
        """
        if not text or not text.strip():
            raise ValueError("Cannot generate embedding for empty text.")

        emb = self.model.encode([text.strip()], normalize_embeddings=True, show_progress_bar=False)
        return np.array(emb, dtype=np.float32)

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """
        Generates normalized embedding vectors for a batch of texts.
        
        Args:
            texts: List of text strings.
            
        Returns:
            np.ndarray of shape (N, dimension), float32.
        """
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        cleaned_texts = [t.strip() if t and t.strip() else " " for t in texts]
        emb = self.model.encode(cleaned_texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
        return np.array(emb, dtype=np.float32)
