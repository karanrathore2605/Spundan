from dataclasses import dataclass


@dataclass
class Document:
    content: str
    source: str


@dataclass
class Chunk:
    chunk_id: int
    content: str
    source: str


@dataclass
class RetrievedChunk:
    chunk_id: int
    content: str
    source: str
    score: float