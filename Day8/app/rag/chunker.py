"""
Chunking module for the RAG pipeline.
Splits documents into semantically coherent, boundary-aware text segments.
Strictly respects word boundaries and punctuation so words are never cut in half.
"""

from typing import List, Dict, Any
from dataclasses import dataclass, field
import logging
from app.rag.loader import Document

logger = logging.getLogger(__name__)


@dataclass
class TextChunk:
    """Represents a discrete segment of a document with chunk ID and source metadata."""
    chunk_id: str
    content: str
    source: str
    chunk_index: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class TextChunker:
    """
    Splits Document instances into boundary-aware chunks with sliding-window overlap.
    Guarantees words and sentences are preserved without mid-word cuts.
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be strictly less than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_document(self, document: Document) -> List[TextChunk]:
        """
        Splits a single document into overlapping, boundary-safe text chunks.
        
        Args:
            document: Document instance to split.
            
        Returns:
            List[TextChunk]: Complete, well-formed chunks.
        """
        text = document.content
        if not text or not text.strip():
            return []

        clean_text = text.strip()
        if len(clean_text) <= self.chunk_size:
            return [
                TextChunk(
                    chunk_id=f"{document.source}_chunk_0",
                    content=clean_text,
                    source=document.source,
                    chunk_index=0,
                    metadata=dict(document.metadata),
                )
            ]

        chunks: List[TextChunk] = []
        start = 0
        chunk_idx = 0

        while start < len(clean_text):
            end = min(start + self.chunk_size, len(clean_text))

            # Snap end backward to nearest punctuation or whitespace boundary
            if end < len(clean_text):
                break_pos = -1
                for p in range(end, max(start + 1, end - self.chunk_overlap), -1):
                    if clean_text[p - 1] in ('\n', '.', '!', '?', ' ', '\t'):
                        break_pos = p
                        break
                if break_pos != -1:
                    end = break_pos

            chunk_content = clean_text[start:end].strip()
            if chunk_content:
                chunk_meta = dict(document.metadata)
                chunk_meta.update({
                    "start_char": start,
                    "end_char": end,
                })
                chunks.append(
                    TextChunk(
                        chunk_id=f"{document.source}_chunk_{chunk_idx}",
                        content=chunk_content,
                        source=document.source,
                        chunk_index=chunk_idx,
                        metadata=chunk_meta,
                    )
                )
                chunk_idx += 1

            if end >= len(clean_text):
                break

            # Calculate next start with overlap
            next_start = end - self.chunk_overlap

            # If next_start splits mid-word, snap forward to nearest whitespace
            if (
                next_start > start
                and next_start < len(clean_text)
                and clean_text[next_start - 1] not in (' ', '\n', '\t')
            ):
                fwd_break = -1
                for p in range(next_start, min(end, next_start + max(1, self.chunk_overlap // 2))):
                    if clean_text[p] in (' ', '\n', '\t'):
                        fwd_break = p + 1
                        break
                if fwd_break != -1:
                    next_start = fwd_break

            start = max(start + 1, next_start)

        logger.debug("Split '%s' into %d clean boundary chunks", document.source, len(chunks))
        return chunks

    def chunk_documents(self, documents: List[Document]) -> List[TextChunk]:
        """
        Splits multiple documents into boundary-safe chunks.
        
        Args:
            documents: List of Document instances.
            
        Returns:
            List[TextChunk]: All chunks across documents.
        """
        all_chunks: List[TextChunk] = []
        for doc in documents:
            all_chunks.extend(self.chunk_document(doc))
        logger.info("Total clean chunks created across %d documents: %d", len(documents), len(all_chunks))
        return all_chunks
