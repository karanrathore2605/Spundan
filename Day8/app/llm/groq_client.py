"""
Groq LLM Client module.
Provides interface to Groq cloud API for inference and grounded RAG answer generation.
"""

from typing import Optional, List, Dict, Any
import logging
from groq import Groq
from app.config import settings

logger = logging.getLogger(__name__)


class GroqClient:
    """Manages interactions with the Groq API."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or settings.groq_api_key
        self.model = model or settings.groq_model
        self._client: Optional[Groq] = None

    @property
    def is_configured(self) -> bool:
        """Returns True if a valid API key appears to be configured."""
        return bool(self.api_key and not self.api_key.startswith("gsk_your_groq_api_key"))

    @property
    def client(self) -> Groq:
        """Initializes and returns the Groq client instance."""
        if not self.is_configured:
            raise ValueError(
                "GROQ_API_KEY is not configured or contains placeholder value. "
                "Please set GROQ_API_KEY in your .env file."
            )
        if self._client is None:
            self._client = Groq(api_key=self.api_key)
        return self._client

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> str:
        """
        Sends chat completion request to Groq API.
        
        Args:
            messages: List of message dicts (role, content).
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in response.
            
        Returns:
            str: Generated completion content.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,  # type: ignore
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.error("Groq chat completion failed: %s", str(e))
            raise RuntimeError(f"Groq API error: {str(e)}") from e

    def generate_grounded_answer(self, query: str, context: str) -> str:
        """
        Synthesizes a grounded answer using retrieved RAG context.
        
        Args:
            query: The user's original question.
            context: Retrieved document chunks with sources.
            
        Returns:
            str: Grounded answer with source citations.
        """
        system_prompt = (
            "You are an expert AI assistant. Answer the user's question using ONLY the provided "
            "context. If the answer cannot be determined from the context, state that clearly. "
            "Always cite the source files mentioned in the context."
        )
        user_message = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
        return self.chat_completion(messages)
