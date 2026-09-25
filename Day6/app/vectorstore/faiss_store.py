import faiss
import numpy as np

from app.models import Chunk


class FAISSVectorStore:

    def __init__(self):
        self.index = None
        self.chunks: list[Chunk] = []

    def build(
        self,
        embeddings: np.ndarray,
        chunks: list[Chunk]
    ):

        if len(embeddings) != len(chunks):
            raise ValueError(
                "Number of embeddings must match number of chunks."
            )

        if len(embeddings) == 0:
            raise ValueError(
                "Cannot build FAISS index with no embeddings."
            )

        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatIP(dimension)

        self.index.add(embeddings)

        self.chunks = chunks

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 3
    ):

        if self.index is None:
            raise RuntimeError(
                "FAISS index has not been built."
            )

        top_k = min(top_k, len(self.chunks))

        scores, indices = self.index.search(
            query_embedding,
            top_k
        )

        results = []

        for score, index in zip(
            scores[0],
            indices[0]
        ):

            if index == -1:
                continue

            chunk = self.chunks[index]

            results.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "content": chunk.content,
                    "source": chunk.source,
                    "score": float(score)
                }
            )

        return results