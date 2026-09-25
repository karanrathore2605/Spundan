"""
Tests for Document Search / RAG Tool.
Verifies document loading, vector similarity search, structured results,
and error handling.
"""

import pytest
from pathlib import Path
from app.rag.loader import DocumentLoader, Document
from app.rag.chunker import TextChunker
from app.rag.retriever import RAGRetriever
from app.tools.document_search import DocumentSearchTool, document_search
from app.config import settings
from app.schemas.tool_schema import ToolResult, SearchResponse


class TestRAGComponents:
    """Tests lower-level RAG components: loader and chunker."""

    def test_document_loader_loads_sample_files(self):
        loader = DocumentLoader(settings.documents_dir)
        docs = loader.load()
        assert len(docs) > 0
        sources = [d.source for d in docs]
        assert "ai.txt" in sources
        assert "sample.txt" in sources

    def test_chunker_creates_overlapping_chunks(self):
        doc = Document(
            content="A" * 500,
            source="test.txt",
            metadata={"file_path": "test.txt"},
        )
        chunker = TextChunker(chunk_size=200, chunk_overlap=50)
        chunks = chunker.chunk_document(doc)
        assert len(chunks) >= 3
        assert chunks[0].source == "test.txt"
        assert chunks[0].chunk_index == 0


class TestDocumentSearchTool:
    """Tests for the DocumentSearchTool class and standalone document_search function."""

    @pytest.fixture(scope="class")
    @classmethod
    def tool(cls):
        """Create shared tool fixture with indexed documents."""
        t = DocumentSearchTool()
        # Pre-build index
        t.retriever.build_index()
        return t

    def test_tool_metadata(self, tool):
        assert tool.name == "document_search"
        assert "document" in tool.description.lower()
        schema = tool.schema
        assert schema.name == "document_search"
        assert "query" in schema.parameters["properties"]

    def test_search_valid_query_explain_ai(self, tool):
        result: ToolResult = tool.execute("Explain AI")
        assert result.success is True
        assert result.error is None
        assert isinstance(result.data, dict)

        results = result.data.get("results", [])
        assert len(results) > 0
        first_match = results[0]
        assert "content" in first_match
        assert "source" in first_match
        assert "score" in first_match
        # Top match should be from ai.txt
        assert first_match["source"] == "ai.txt"
        assert "artificial intelligence" in first_match["content"].lower()

        # Test output formatting
        formatted = tool.format_output(result)
        assert "Retrieved Context:" in formatted
        assert "ai.txt" in formatted
        assert "Sources:" in formatted

    def test_search_machine_learning_query(self, tool):
        result: ToolResult = tool.execute("What is machine learning?")
        assert result.success is True
        results = result.data.get("results", [])
        assert len(results) > 0
        assert results[0]["source"] == "machine_learning.txt"

    def test_search_empty_query(self, tool):
        result: ToolResult = tool.execute("")
        assert result.success is False
        assert "empty" in result.error.lower()

        whitespace_result: ToolResult = tool.execute("   ")
        assert whitespace_result.success is False

    def test_standalone_document_search_function(self, tool):
        data = document_search("Explain AI", retriever=tool.retriever)
        assert isinstance(data, dict)
        assert data["query"] == "Explain AI"
        assert len(data["results"]) > 0
