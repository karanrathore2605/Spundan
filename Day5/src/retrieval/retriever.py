from src.embeddings.embedding_service import EmbeddingService
from src.models import RetrievedChunk
from src.vectorstore.faiss_store import FAISSStore


class Retriever:
    """
    Retrieves relevant document chunks for a user query.
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: FAISSStore,
        top_k: int = 5,
    ):
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.top_k = top_k

    def retrieve(
        self,
        query: str,
    ) -> list[RetrievedChunk]:

        query_embedding = (
            self.embedding_service
            .generate_query_embedding(query)
        )

        results = self.vector_store.search(
            query_embedding,
            top_k=self.top_k,
        )

        return results