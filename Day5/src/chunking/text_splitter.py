import re

from src.models import Document, DocumentChunk


class TextSplitter:
    """
    Splits documents into overlapping text chunks.
    """

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 100,
        min_chunk_size: int = 50,
    ):
        if chunk_overlap >= chunk_size:
            raise ValueError(
                "chunk_overlap must be smaller than chunk_size."
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

    def split_documents(
        self,
        documents: list[Document],
    ) -> list[DocumentChunk]:

        chunks: list[DocumentChunk] = []

        chunk_id = 0

        for document in documents:

            document_chunks = self._split_text(document.text)

            for text in document_chunks:

                if len(text.strip()) < self.min_chunk_size:
                    continue

                chunks.append(
                    DocumentChunk(
                        text=text,
                        source=document.source,
                        page=document.page,
                        chunk_id=chunk_id,
                    )
                )

                chunk_id += 1

        return chunks

    def _split_text(self, text: str) -> list[str]:
        """
        Split text approximately by sentences while maintaining
        the configured chunk size and overlap.
        """

        sentences = re.split(
            r"(?<=[.!?])\s+",
            text,
        )

        sentences = [
            sentence.strip()
            for sentence in sentences
            if sentence.strip()
        ]

        chunks: list[str] = []

        current_chunk: list[str] = []
        current_length = 0

        for sentence in sentences:

            sentence_length = len(sentence)

            if (
                current_length + sentence_length
                <= self.chunk_size
            ):
                current_chunk.append(sentence)

                current_length += sentence_length + 1

            else:

                if current_chunk:
                    chunks.append(
                        " ".join(current_chunk)
                    )

                overlap_text = self._get_overlap(
                    current_chunk
                )

                current_chunk = []

                if overlap_text:
                    current_chunk.append(overlap_text)

                current_chunk.append(sentence)

                current_length = len(
                    " ".join(current_chunk)
                )

        if current_chunk:
            chunks.append(
                " ".join(current_chunk)
            )

        return chunks

    def _get_overlap(
        self,
        sentences: list[str],
    ) -> str:

        if not sentences:
            return ""

        overlap_sentences = []

        length = 0

        for sentence in reversed(sentences):

            if (
                length + len(sentence)
                > self.chunk_overlap
            ):
                break

            overlap_sentences.insert(
                0,
                sentence,
            )

            length += len(sentence)

        return " ".join(overlap_sentences)