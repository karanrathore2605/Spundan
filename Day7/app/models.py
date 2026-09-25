from dataclasses import dataclass, field
from typing import Any


@dataclass
class Document:
    """Represents a loaded text document."""
    content: str
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.metadata:
            self.metadata = {"source": self.source}
        elif "source" not in self.metadata:
            self.metadata["source"] = self.source


@dataclass
class Chunk:
    """Represents a text chunk split from a document with full metadata."""
    content: str
    chunk_id: int
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.metadata:
            self.metadata = {
                "chunk_id": self.chunk_id,
                "source": self.source,
                "text": self.content,
            }
        else:
            self.metadata.setdefault("chunk_id", self.chunk_id)
            self.metadata.setdefault("source", self.source)
            self.metadata.setdefault("text", self.content)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "source": self.source,
            "content": self.content,
            "metadata": self.metadata,
        }


@dataclass
class RetrievedChunk:
    """Represents a chunk retrieved from the vector store with similarity score."""
    chunk_id: int
    content: str
    source: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "source": self.source,
            "content": self.content,
            "score": self.score,
            "metadata": self.metadata,
        }