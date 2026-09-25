from app.config import TOP_K
from app.embeddings.embedding_service import EmbeddingService
from app.vectorstore.faiss_store import FAISSVectorStore


class Retriever:

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: FAISSVectorStore
    ):
        self.embedding_service = embedding_service
        self.vector_store = vector_store

    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K
    ):

        query_embedding = (
            self.embedding_service
            .generate_query_embedding(query)
        )

        return self.vector_store.search(
            query_embedding,
            top_k
        )