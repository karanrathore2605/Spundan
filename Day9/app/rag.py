from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
import pickle

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from .config import DOCUMENTS_DIR, VECTOR_STORE_DIR, EMBEDDING_MODEL, TOP_K
from .llm import generate_answer

@dataclass
class Chunk:
    text: str
    source: str

class RAGEngine:
    def __init__(self) -> None:
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.chunks: List[Chunk] = []
        self.index = None
        self._load_or_build()

    def _read_documents(self) -> List[Chunk]:
        DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
        chunks: List[Chunk] = []
        for path in sorted(DOCUMENTS_DIR.glob("*.txt")):
            text = path.read_text(encoding="utf-8").strip()
            if not text:
                continue
            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
            if not paragraphs:
                paragraphs = [text]
            for paragraph in paragraphs:
                words = paragraph.split()
                size, overlap = 180, 30
                start = 0
                while start < len(words):
                    piece = " ".join(words[start:start + size]).strip()
                    if piece:
                        chunks.append(Chunk(piece, path.name))
                    if start + size >= len(words):
                        break
                    start += size - overlap
        return chunks

    def _load_or_build(self) -> None:
        VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
        index_path = VECTOR_STORE_DIR / "index.faiss"
        chunks_path = VECTOR_STORE_DIR / "chunks.pkl"
        current_files = list(DOCUMENTS_DIR.glob("*.txt"))
        if index_path.exists() and chunks_path.exists() and current_files:
            try:
                self.index = faiss.read_index(str(index_path))
                self.chunks = pickle.loads(chunks_path.read_bytes())
                return
            except Exception:
                pass
        self.rebuild()

    def rebuild(self) -> None:
        self.chunks = self._read_documents()
        if not self.chunks:
            self.index = None
            return
        vectors = self.model.encode(
            [c.text for c in self.chunks],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)
        faiss.write_index(self.index, str(VECTOR_STORE_DIR / "index.faiss"))
        (VECTOR_STORE_DIR / "chunks.pkl").write_bytes(pickle.dumps(self.chunks))

    def retrieve(self, query: str) -> List[Tuple[Chunk, float]]:
        if self.index is None or not self.chunks:
            return []
        vector = self.model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        ).astype("float32")
        scores, ids = self.index.search(vector, min(TOP_K, len(self.chunks)))
        return [(self.chunks[i], float(score)) for i, score in zip(ids[0], scores[0]) if i >= 0]

    def answer(self, query: str) -> dict:
        results = self.retrieve(query)
        if not results:
            return {
                "success": False,
                "answer": "I could not find any documents to answer this question from.",
                "sources": [],
            }

        context = "\n\n".join(
            f"[Source: {chunk.source}]\n{chunk.text}" for chunk, _ in results
        )
        prompt = f"""Answer the question using only the supplied context.
If the context is insufficient, say that the documents do not contain enough information.
Do not invent facts.

Context:
{context}

Question:
{query}
"""
        answer = generate_answer(prompt)
        sources = list(dict.fromkeys(chunk.source for chunk, _ in results))
        return {"success": True, "answer": answer, "sources": sources}

_engine = None

def get_rag_engine() -> RAGEngine:
    global _engine
    if _engine is None:
        _engine = RAGEngine()
    return _engine

def rag_tool(query: str) -> dict:
    return get_rag_engine().answer(query)
