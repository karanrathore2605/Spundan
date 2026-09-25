import time

from google import genai
from google.genai import types

from src.config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
)


class GeminiService:
    """
    Service responsible for communication with Gemini.
    """

    def __init__(
        self,
        model_name: str = GEMINI_MODEL,
    ):
        self.model_name = model_name

        self.client = genai.Client(
            api_key=GEMINI_API_KEY
        )

    def generate_answer(
        self,
        prompt: str,
    ) -> str:

        if not prompt.strip():
            raise ValueError(
                "Prompt cannot be empty."
            )

        max_retries = 3

        for attempt in range(max_retries):

            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.2,
                        max_output_tokens=1024,
                    ),
                )

                if not response.text:
                    raise RuntimeError(
                        "Gemini returned an empty response."
                    )

                return response.text.strip()

            except Exception as exc:

                error_text = str(exc)

                is_retryable = (
                    "503" in error_text
                    or "UNAVAILABLE" in error_text
                    or "429" in error_text
                )

                if not is_retryable:
                    raise

                if attempt == max_retries - 1:
                    raise RuntimeError(
                        "Gemini is temporarily unavailable "
                        "after multiple retries."
                    ) from exc

                delay = 2 ** attempt

                print(
                    f"Gemini temporarily unavailable. "
                    f"Retrying in {delay} seconds..."
                )

                time.sleep(delay)