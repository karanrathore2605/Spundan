from pathlib import Path
import os

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")


# Directories
DOCUMENTS_DIR = BASE_DIR / "documents"


# Chunking
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


# Retrieval
TOP_K = 4
SIMILARITY_THRESHOLD = 0.35


# Embedding model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# Groq
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "llama-3.3-70b-versatile"
)