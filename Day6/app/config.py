import os
from pathlib import Path

from dotenv import load_dotenv


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()


# ============================================================
# PROJECT DIRECTORIES
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DOCUMENTS_DIR = BASE_DIR / "data" / "documents"
STORAGE_DIR = BASE_DIR / "storage"


# ============================================================
# GEMINI API CONFIGURATION
# ============================================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY is missing. Add GROQ_API_KEY to your .env file."
    )


# ============================================================
# MODELS
# ============================================================

# Used for converting text into embeddings
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Used for generating the final RAG answer
GEMINI_MODEL = "gemini-3.8-flash"


# ============================================================
# RAG CONFIGURATION
# ============================================================

# Number of chunks retrieved from FAISS
TOP_K = 3

# Number of characters in each chunk
CHUNK_SIZE = 500

# Number of overlapping characters between chunks
CHUNK_OVERLAP = 100


# ============================================================
# GEMINI RETRY CONFIGURATION
# ============================================================

# Maximum number of attempts when Gemini temporarily fails
MAX_RETRIES = 3

# Initial retry delay in seconds
INITIAL_RETRY_DELAY = 2