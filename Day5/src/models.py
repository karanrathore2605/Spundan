from dataclasses import dataclass
from typing import Optional


@dataclass
class Document:
    """
    Represents a loaded document.
    """

    text: str
    source: str
    page: Optional[int] = None


@dataclass
class DocumentChunk:
    """
    Represents a chunk created from a document.
    """

    text: str
    source: str
    page: Optional[int] = None
    chunk_id: Optional[int] = None


@dataclass
class RetrievedChunk:
    """
    Represents a chunk returned by vector similarity search.
    """

    chunk: DocumentChunk
    score: float


@dataclass
class RAGResponse:
    """
    Represents the final RAG response.
    """

    answer: str
    sources: list[RetrievedChunk]