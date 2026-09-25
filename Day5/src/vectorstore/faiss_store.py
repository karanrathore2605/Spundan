import json
from pathlib import Path

import faiss
import numpy as np

from src.models import DocumentChunk, RetrievedChunk


class FAISSStore:
    """
    Vector store based on FAISS.
    """

    def __init__(self):
        self.index = None
        self.chunks: list[DocumentChunk] = []

    def build(
        self,
        embeddings: np.ndarray,
        chunks: list[DocumentChunk],
    ) -> None:

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
        top_k: int = 5,
    ) -> list[RetrievedChunk]:

        if self.index is None:
            raise RuntimeError(
                "FAISS index has not been built."
            )

        if not self.chunks:
            return []

        top_k = min(
            top_k,
            len(self.chunks),
        )

        scores, indices = self.index.search(
            query_embedding,
            top_k,
        )

        results: list[RetrievedChunk] = []

        for score, index in zip(
            scores[0],
            indices[0],
        ):

            if index < 0:
                continue

            results.append(
                RetrievedChunk(
                    chunk=self.chunks[index],
                    score=float(score),
                )
            )

        return results

    def save(
        self,
        index_path: Path,
        metadata_path: Path,
    ) -> None:

        if self.index is None:
            raise RuntimeError(
                "Cannot save an empty FAISS index."
            )

        index_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        faiss.write_index(
            self.index,
            str(index_path),
        )

        metadata = [
            {
                "text": chunk.text,
                "source": chunk.source,
                "page": chunk.page,
                "chunk_id": chunk.chunk_id,
            }
            for chunk in self.chunks
        ]

        metadata_path.write_text(
            json.dumps(
                metadata,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    def load(
        self,
        index_path: Path,
        metadata_path: Path,
    ) -> None:

        if not index_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found: {index_path}"
            )

        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata file not found: {metadata_path}"
            )

        self.index = faiss.read_index(
            str(index_path)
        )

        metadata = json.loads(
            metadata_path.read_text(
                encoding="utf-8"
            )
        )

        self.chunks = [
            DocumentChunk(
                text=item["text"],
                source=item["source"],
                page=item.get("page"),
                chunk_id=item.get("chunk_id"),
            )
            for item in metadata
        ]