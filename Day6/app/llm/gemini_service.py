import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()


class GeminiService:

    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError(
                "GROQ_API_KEY is missing from .env"
            )

        self.client = Groq(api_key=api_key)

        self.model = "openai/gpt-oss-120b"

    def generate(self, prompt: str) -> str:
        """
        Generate an answer using Groq.
        This method name is kept as 'generate'
        because main.py already expects it.
        """

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.2,
            max_tokens=500
        )

        return response.choices[0].message.content