import logging
from typing import Any
import groq
from groq import Groq

from app.config import (
    GROQ_API_KEY,
    GROQ_MODEL,
    MAX_OUTPUT_TOKENS,
    TEMPERATURE,
    validate_groq_api_key,
)

logger = logging.getLogger(__name__)


class GroqService:
    """Manages LLM completions via Groq API with robust error handling."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = GROQ_MODEL,
        temperature: float = TEMPERATURE,
        max_output_tokens: int = MAX_OUTPUT_TOKENS,
    ):
        # Edge Case 6: Validate GROQ_API_KEY
        resolved_key = (api_key or GROQ_API_KEY).strip()
        if not resolved_key:
            try:
                resolved_key = validate_groq_api_key()
            except ValueError as exc:
                raise ValueError(
                    "GROQ_API_KEY is not configured.\n"
                    "Please add it to your .env file."
                ) from exc

        self.client = Groq(api_key=resolved_key)
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens

    def generate(
        self,
        prompt: str,
        system_instruction: str | None = None,
    ) -> str:
        """
        Send prompt to Groq LLM and return generated text.
        Gracefully handles network errors, auth errors, and rate limits without crashing.
        """
        messages: list[dict[str, Any]] = []

        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})

        messages.append({"role": "user", "content": prompt})

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_output_tokens,
            )

            if response.choices and len(response.choices) > 0:
                content = response.choices[0].message.content
                return content.strip() if content else ""

            return ""

        # Edge Case 7: Groq API failure handling
        except groq.AuthenticationError as exc:
            logger.error(f"Groq Authentication failed: {exc}")
            return (
                "Unable to generate the answer because GROQ_API_KEY is invalid.\n"
                "Please verify the API key in your .env file."
            )

        except groq.RateLimitError as exc:
            logger.error(f"Groq Rate limit exceeded: {exc}")
            return (
                "Unable to generate the answer because the LLM service rate limit was exceeded.\n"
                "Please wait a moment before trying again."
            )

        except groq.APIConnectionError as exc:
            logger.error(f"Groq Connection error: {exc}")
            return (
                "Unable to generate the answer because the LLM service "
                "is currently unavailable."
            )

        except groq.APIStatusError as exc:
            logger.error(f"Groq Status error ({exc.status_code}): {exc.message}")
            return (
                "Unable to generate the answer because the LLM service "
                "is currently unavailable."
            )

        except Exception as exc:
            logger.error(f"Unexpected Groq LLM error: {type(exc).__name__}: {exc}")
            return (
                "Unable to generate the answer because the LLM service "
                "is currently unavailable."
            )
