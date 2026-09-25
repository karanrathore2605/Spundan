from typing import Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.config import CHUNK_SIZE, CHUNK_OVERLAP
from app.models import Chunk, Document


class TextChunker:
    """Chunks documents into overlapping text windows while preserving metadata."""

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        overlap: int = CHUNK_OVERLAP,
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=[
                "\n\n",
                "\n",
                ". ",
                "? ",
                "! ",
                " ",
                "",
            ],
            length_function=len,
        )

    def split_document(self, document: Document) -> list[Chunk]:
        """Split a single Document into Chunks with metadata."""
        if not document.content or not document.content.strip():
            return []

        text_chunks = self.splitter.split_text(document.content)
        chunks: list[Chunk] = []

        for chunk_id, text in enumerate(text_chunks, start=1):
            clean_text = text.strip()
            if not clean_text:
                continue

            metadata: dict[str, Any] = {
                "chunk_id": chunk_id,
                "source": document.source,
                "text": clean_text,
            }
            if hasattr(document, "metadata") and isinstance(document.metadata, dict):
                for k, v in document.metadata.items():
                    if k not in metadata:
                        metadata[k] = v

            chunks.append(
                Chunk(
                    content=clean_text,
                    chunk_id=chunk_id,
                    source=document.source,
                    metadata=metadata,
                )
            )

        return chunks

    def split(self, documents: list[Document]) -> list[Chunk]:
        """Split a list of Documents into Chunks with metadata."""
        all_chunks: list[Chunk] = []
        for doc in documents:
            all_chunks.extend(self.split_document(doc))
        return all_chunks

    def create_chunks(self, documents: list[Document]) -> list[Chunk]:
        """Alias for split to ensure backwards compatibility."""
        return self.split(documents)
