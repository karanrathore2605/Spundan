import logging
from typing import Any
from app.config import TOP_K, SIMILARITY_THRESHOLD, sanitize_top_k
from app.embeddings import EmbeddingService
from app.vector_store import FAISSVectorStore

logger = logging.getLogger(__name__)


class Retriever:
    """Retrieves and filters the most relevant document chunks for a given query."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: FAISSVectorStore,
        similarity_threshold: float = SIMILARITY_THRESHOLD,
    ):
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.similarity_threshold = similarity_threshold

    def retrieve(
        self,
        question: str,
        top_k: int = TOP_K,
    ) -> list[dict[str, Any]]:
        """
        Retrieve chunks matching the query that meet the similarity threshold.
        Gracefully handles empty query, invalid top_k, and zero results.
        """
        # Edge Case 3: Empty question
        if not question or not question.strip():
            return []

        clean_question = question.strip()
        safe_top_k = sanitize_top_k(top_k)

        # Generate query embedding
        try:
            query_embedding = self.embedding_service.generate_query_embedding(clean_question)
        except Exception as exc:
            logger.error(f"Failed to generate query embedding: {exc}")
            return []

        # Search FAISS index
        raw_results = self.vector_store.search(query_embedding, top_k=safe_top_k)
        if not raw_results:
            return []

        # Edge Case 5: Apply similarity threshold
        filtered_results: list[dict[str, Any]] = []
        for res in raw_results:
            score = res.get("score", 0.0)
            if score >= self.similarity_threshold:
                filtered_results.append(res)
            else:
                logger.debug(
                    f"Chunk {res.get('chunk_id')} from {res.get('source')} "
                    f"skipped due to low similarity ({score:.4f} < {self.similarity_threshold})"
                )

        # Deduplicate results by chunk content while preserving order
        unique_results: list[dict[str, Any]] = []
        seen_texts: set[str] = set()

        for chunk in filtered_results:
            text = chunk.get("content", "").strip()
            if not text or text in seen_texts:
                continue
            seen_texts.add(text)
            unique_results.append(chunk)

        return unique_results
