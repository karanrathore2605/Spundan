import os
import warnings

from dotenv import load_dotenv
from google import genai


load_dotenv()

warnings.filterwarnings(
    "ignore",
    message=".*automatic function calling.*"
)

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)


response = client.models.generate_content(
    model="gemini-3.5-flash-lite",
    contents="Explain Python in 3 simple sentences."
)

print(response.text)