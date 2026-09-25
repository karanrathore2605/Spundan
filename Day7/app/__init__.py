"""
Day 7 - Week 1 Unified RAG System Package
"""

import sys

# Ensure UTF-8 output encoding across Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from app.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DOCUMENTS_DIR,
    EMBEDDING_MODEL,
    GROQ_API_KEY,
    GROQ_MODEL,
    MAX_QUERY_LENGTH,
    MAX_OUTPUT_TOKENS,
    SIMILARITY_THRESHOLD,
    STORAGE_DIR,
    TEMPERATURE,
    TOP_K,
)
from app.models import Chunk, Document, RetrievedChunk
from app.loader import DocumentLoader
from app.chunker import TextChunker
from app.embeddings import EmbeddingService
from app.vector_store import FAISSVectorStore
from app.retriever import Retriever
from app.llm import GroqService
from app.prompts import SYSTEM_PROMPT, build_rag_prompt
from app.citation import format_final_response, format_sources
from app.rag_pipeline import RAGPipeline

__all__ = [
    "Chunk",
    "Document",
    "RetrievedChunk",
    "DocumentLoader",
    "TextChunker",
    "EmbeddingService",
    "FAISSVectorStore",
    "Retriever",
    "GroqService",
    "SYSTEM_PROMPT",
    "build_rag_prompt",
    "format_sources",
    "format_final_response",
    "RAGPipeline",
    "TOP_K",
    "CHUNK_SIZE",
    "CHUNK_OVERLAP",
    "SIMILARITY_THRESHOLD",
    "GROQ_API_KEY",
    "GROQ_MODEL",
    "EMBEDDING_MODEL",
    "DOCUMENTS_DIR",
    "STORAGE_DIR",
]
