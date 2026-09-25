import os
from pathlib import Path
from dotenv import load_dotenv

# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables from .env file
ENV_FILE = BASE_DIR / ".env"
if ENV_FILE.exists():
    load_dotenv(ENV_FILE)
else:
    load_dotenv()

# Documents directories (support both documents/ and data/documents/)
DOCUMENTS_DIR = BASE_DIR / "documents"
if not DOCUMENTS_DIR.exists():
    DOCUMENTS_DIR = BASE_DIR / "data" / "documents"

# Storage / Vector Store directory
STORAGE_DIR = BASE_DIR / "data" / "vector_store"
if not STORAGE_DIR.exists():
    STORAGE_DIR = BASE_DIR / "storage"

# ============================================================
# GROQ API CONFIGURATION
# ============================================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()

# Default Groq model (can be overridden via .env)
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b").strip()

# Generation parameters
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "500"))

# ============================================================
# EMBEDDING & VECTOR STORE CONFIGURATION
# ============================================================

# HuggingFace sentence-transformers model
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2").strip()

# ============================================================
# RAG RETRIEVAL CONFIGURATION
# ============================================================

# Chunking settings
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))

# Number of chunks retrieved from FAISS
DEFAULT_TOP_K = 3
try:
    _raw_top_k = int(os.getenv("TOP_K", "3"))
    TOP_K = _raw_top_k if _raw_top_k > 0 else DEFAULT_TOP_K
except (ValueError, TypeError):
    TOP_K = DEFAULT_TOP_K

# Minimum cosine similarity threshold for retrieval
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.35"))

# Maximum question length to guard against buffer/token abuse
MAX_QUERY_LENGTH = int(os.getenv("MAX_QUERY_LENGTH", "1000"))


def sanitize_top_k(top_k: int | None) -> int:
    """Ensure TOP_K is a positive integer, falling back to safe default."""
    if top_k is None:
        return TOP_K
    try:
        val = int(top_k)
        return val if val > 0 else DEFAULT_TOP_K
    except (ValueError, TypeError):
        return DEFAULT_TOP_K


def validate_groq_api_key() -> str:
    """
    Validate that GROQ_API_KEY is present.
    Raises ValueError with a user-friendly message if missing.
    """
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is not configured.\n"
            "Please add it to your .env file."
        )
    return api_key
