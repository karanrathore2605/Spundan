from dataclasses import dataclass, field
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import CHUNK_SIZE, CHUNK_OVERLAP


@dataclass
class Chunk:
    """
    Represents one chunk of a document.
    """

    content: str

    # Direct attributes expected by the retrieval layer
    chunk_id: int = 0

    # Source/document information
    source: str = ""

    # Additional metadata
    metadata: dict[str, Any] = field(default_factory=dict)


class TextChunker:

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        overlap: int = CHUNK_OVERLAP,
    ):

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

    def split(self, documents) -> list[Chunk]:

        all_chunks = []

        for document in documents:

            # ==================================================
            # GET DOCUMENT TEXT
            # ==================================================

            if hasattr(document, "content"):

                text = document.content

            elif isinstance(document, dict):

                text = document.get(
                    "content",
                    ""
                )

            elif isinstance(document, str):

                text = document

            else:

                continue

            if not text or not text.strip():
                continue

            # ==================================================
            # GET DOCUMENT METADATA
            # ==================================================

            if hasattr(document, "metadata"):

                metadata = document.metadata or {}

            elif isinstance(document, dict):

                metadata = document.get(
                    "metadata",
                    {}
                )

            else:

                metadata = {}

            metadata = dict(metadata)

            # ==================================================
            # GET SOURCE
            # ==================================================

            source = (
                metadata.get("source")
                or metadata.get("filename")
                or metadata.get("file_name")
                or ""
            )

            # ==================================================
            # SPLIT DOCUMENT
            # ==================================================

            document_chunks = (
                self.splitter.split_text(text)
            )

            # ==================================================
            # CREATE CHUNK OBJECTS
            # ==================================================

            for chunk_id, chunk_text in enumerate(
                document_chunks,
                start=1,
            ):

                chunk_text = chunk_text.strip()

                if not chunk_text:
                    continue

                chunk_metadata = metadata.copy()

                chunk_metadata["chunk_id"] = chunk_id
                chunk_metadata["source"] = source

                chunk = Chunk(
                    content=chunk_text,
                    chunk_id=chunk_id,
                    source=source,
                    metadata=chunk_metadata,
                )

                all_chunks.append(chunk)

        return all_chunks

    def split_text(self, text: str) -> list[str]:

        if not text or not text.strip():
            return []

        return self.splitter.split_text(text)

    def create_chunks(self, documents) -> list[Chunk]:

        return self.split(documents)