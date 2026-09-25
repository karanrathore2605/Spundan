from app.config import TOP_K
from app.llm.gemini_service import GeminiService
from app.rag.citation import build_final_response
from app.rag.prompt import build_prompt
from app.retrieval.retriever import Retriever


class RAGPipeline:

    def __init__(
        self,
        retriever: Retriever,
        gemini_service: GeminiService
    ):
        self.retriever = retriever
        self.gemini_service = gemini_service

    def ask(
        self,
        question: str,
        top_k: int = TOP_K
    ) -> str:

        question = question.strip()

        if not question:
            raise ValueError(
                "Question cannot be empty."
            )

        # Step 1: Retrieve relevant chunks
        retrieved_chunks = (
            self.retriever.retrieve(
                question,
                top_k
            )
        )

        if not retrieved_chunks:
            return (
                "Answer:\n"
                "I could not find relevant information "
                "in the provided documents.\n\n"
                "Sources:\n"
                "- No relevant sources found."
            )

        # Step 2: Build grounded prompt
        prompt = build_prompt(
            question,
            retrieved_chunks
        )

        # Step 3: Generate answer
        answer = self.gemini_service.generate(
            prompt
        )

        # Step 4: Attach sources
        return build_final_response(
            answer,
            retrieved_chunks
        )