import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_MODEL


class EmbeddingService:
    """
    Generates vector embeddings using Sentence Transformers.
    """

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL,
    ):
        self.model = SentenceTransformer(model_name)

    def generate_embeddings(
        self,
        texts: list[str],
    ) -> np.ndarray:

        if not texts:
            raise ValueError(
                "Cannot generate embeddings for empty text list."
            )

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        return embeddings.astype("float32")

    def generate_query_embedding(
        self,
        query: str,
    ) -> np.ndarray:

        if not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        return embedding.astype("float32")