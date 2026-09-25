import logging
import numpy as np
from sentence_transformers import SentenceTransformer
from app.config import EMBEDDING_MODEL

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Generates normalized dense embeddings using sentence-transformers."""

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model_name = model_name
        self._model = None

    @property
    def model(self) -> SentenceTransformer:
        """Lazy load the sentence transformer model."""
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def generate_embeddings(self, texts: list[str]) -> np.ndarray:
        """
        Generate L2-normalized embeddings for a list of strings.
        Returns float32 ndarray of shape (len(texts), embedding_dim).
        """
        if not texts:
            raise ValueError("Cannot generate embeddings for an empty list of texts.")

        clean_texts = [t.strip() for t in texts if t and t.strip()]
        if not clean_texts:
            raise ValueError("All provided texts were empty.")

        embeddings = self.model.encode(
            clean_texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.astype("float32")

    def generate_query_embedding(self, query: str) -> np.ndarray:
        """
        Generate L2-normalized embedding for a single user query.
        Returns float32 ndarray of shape (1, embedding_dim).
        """
        clean_query = query.strip()
        if not clean_query:
            raise ValueError("Query cannot be empty.")

        embedding = self.model.encode(
            [clean_query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embedding.astype("float32")
