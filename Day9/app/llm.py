from groq import Groq
from .config import GROQ_API_KEY, GROQ_MODEL

_client = None

def _get_client() -> Groq:
    global _client
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is missing. Add it to .env.")
    if _client is None:
        _client = Groq(api_key=GROQ_API_KEY)
    return _client

def generate_answer(prompt: str) -> str:
    response = _get_client().chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": "You are a concise, helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()
