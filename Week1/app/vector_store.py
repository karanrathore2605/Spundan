import json
import logging
from pathlib import Path
from typing import Any
import faiss
import numpy as np

from app.config import DEFAULT_TOP_K, sanitize_top_k
from app.models import Chunk

logger = logging.getLogger(__name__)


class FAISSVectorStore:
    """Manages FAISS similarity index and associated chunk metadata with persistence."""

    def __init__(self):
        self.index: faiss.IndexFlatIP | None = None
        self.chunks: list[Chunk] = []

    @property
    def is_ready(self) -> bool:
        """Check if index is built, has items, and chunks metadata is synced."""
        return (
            self.index is not None
            and self.index.ntotal > 0
            and len(self.chunks) == self.index.ntotal
        )

    def build(self, embeddings: np.ndarray, chunks: list[Chunk]) -> None:
        """Build a new FAISS index from scratch with embeddings and chunks."""
        if len(embeddings) != len(chunks):
            raise ValueError(
                f"Mismatch: {len(embeddings)} embeddings vs {len(chunks)} chunks."
            )
        if len(embeddings) == 0:
            raise ValueError("Cannot build FAISS index with zero embeddings.")

        embeddings = np.ascontiguousarray(embeddings, dtype="float32")
        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)
        self.chunks = list(chunks)

    def add(self, embeddings: np.ndarray, chunks: list[Chunk]) -> None:
        """Add incremental embeddings and chunks to an existing index."""
        if len(embeddings) != len(chunks):
            raise ValueError("Number of embeddings must match number of chunks.")
        if len(embeddings) == 0:
            return

        embeddings = np.ascontiguousarray(embeddings, dtype="float32")

        if self.index is None:
            self.build(embeddings, chunks)
        else:
            if embeddings.shape[1] != self.index.d:
                raise ValueError(
                    f"Embedding dimension {embeddings.shape[1]} does not match index dimension {self.index.d}."
                )
            self.index.add(embeddings)
            self.chunks.extend(chunks)

    def search(self, query_embedding: np.ndarray, top_k: int = DEFAULT_TOP_K) -> list[dict[str, Any]]:
        """
        Search for top_k most similar chunks.
        Safely validates top_k and checks index state.
        """
        safe_top_k = sanitize_top_k(top_k)

        if not self.is_ready:
            logger.warning("FAISS search called on uninitialized or empty index.")
            return []

        # Bound top_k to actual number of chunks
        effective_top_k = min(safe_top_k, len(self.chunks))
        if effective_top_k <= 0:
            return []

        query_embedding = np.ascontiguousarray(query_embedding, dtype="float32")
        if query_embedding.ndim == 1:
            query_embedding = np.expand_dims(query_embedding, axis=0)

        scores, indices = self.index.search(query_embedding, effective_top_k)

        results: list[dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1 or idx >= len(self.chunks):
                continue

            chunk = self.chunks[idx]
            results.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "source": chunk.source,
                    "content": chunk.content,
                    "score": float(score),
                    "metadata": chunk.metadata,
                }
            )

        return results

    def save(self, storage_dir: Path | str) -> None:
        """Save FAISS index and chunk metadata to disk."""
        target_dir = Path(storage_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        index_file = target_dir / "index.faiss"
        metadata_file = target_dir / "chunks.json"

        if self.index is not None:
            faiss.write_index(self.index, str(index_file))

        chunk_data = [chunk.to_dict() for chunk in self.chunks]
        metadata_file.write_text(json.dumps(chunk_data, indent=2), encoding="utf-8")

    def load(self, storage_dir: Path | str) -> bool:
        """
        Load FAISS index and chunk metadata from disk.
        Returns True if successfully loaded and valid, False if missing or corrupted.
        """
        target_dir = Path(storage_dir)
        index_file = target_dir / "index.faiss"
        metadata_file = target_dir / "chunks.json"

        if not index_file.exists() or not metadata_file.exists():
            return False

        try:
            loaded_index = faiss.read_index(str(index_file))
            raw_metadata = json.loads(metadata_file.read_text(encoding="utf-8"))

            loaded_chunks = [
                Chunk(
                    content=item["content"],
                    chunk_id=item["chunk_id"],
                    source=item["source"],
                    metadata=item.get("metadata", {}),
                )
                for item in raw_metadata
            ]

            if loaded_index.ntotal != len(loaded_chunks):
                logger.warning(
                    f"Corrupted vector store: index has {loaded_index.ntotal} entries "
                    f"but metadata has {len(loaded_chunks)} items."
                )
                return False

            self.index = loaded_index
            self.chunks = loaded_chunks
            return True

        except Exception as exc:
            logger.warning(f"Failed to load FAISS index from disk ({exc}). Index will need rebuild.")
            return False
