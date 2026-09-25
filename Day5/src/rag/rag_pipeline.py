from src.llm.gemini_service import GeminiService
from src.models import RAGResponse
from src.rag.prompt import build_rag_prompt
from src.retrieval.retriever import Retriever


class RAGPipeline:
    """
    Orchestrates retrieval and LLM generation.
    """

    def __init__(
        self,
        retriever: Retriever,
        llm: GeminiService,
    ):
        self.retriever = retriever
        self.llm = llm

    def ask(
        self,
        question: str,
    ) -> RAGResponse:

        question = question.strip()

        if not question:
            raise ValueError(
                "Question cannot be empty."
            )

        # -------------------------------------------------
        # Step 1: Retrieve relevant chunks
        # -------------------------------------------------

        retrieved_chunks = (
            self.retriever.retrieve(question)
        )

        if not retrieved_chunks:
            return RAGResponse(
                answer=(
                    "I could not find relevant information "
                    "in the provided documents."
                ),
                sources=[],
            )

        # -------------------------------------------------
        # Step 2: Build grounded prompt
        # -------------------------------------------------

        prompt = build_rag_prompt(
            question=question,
            retrieved_chunks=retrieved_chunks,
        )

        # -------------------------------------------------
        # Step 3: Generate answer with Gemini
        # -------------------------------------------------

        answer = self.llm.generate_answer(
            prompt
        )

        # -------------------------------------------------
        # Step 4: Return answer + sources
        # -------------------------------------------------

        return RAGResponse(
            answer=answer,
            sources=retrieved_chunks,
        )