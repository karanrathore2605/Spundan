"""
Configuration module for Day 8 LLM Tools application.
Loads settings from environment variables and sets up application-wide logging.
"""

import os
import logging
from pathlib import Path
from dataclasses import dataclass
from dotenv import load_dotenv

# Base Directory: Day8 root folder
BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables from .env file if present
load_dotenv(BASE_DIR / ".env")


@dataclass(frozen=True)
class AppConfig:
    """Application configuration parameters."""
    
    # Base paths
    base_dir: Path = BASE_DIR
    documents_dir: Path = BASE_DIR / "documents"
    
    # LLM Settings (Groq)
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_model: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    
    # Embedding Model Settings (Sentence Transformers)
    embedding_model_name: str = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
    
    # RAG Retrieval Settings
    top_k_results: int = int(os.getenv("TOP_K_RESULTS", "3"))
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "500"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "100"))
    
    # Logging
    log_level: str = os.getenv("LOG_LEVEL", "INFO")


# Singleton instance
settings = AppConfig()


def setup_logging(level: str = settings.log_level) -> None:
    """Configures structured application logging."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# Initialize logging when module is imported
setup_logging()
logger = logging.getLogger("day8_app")
