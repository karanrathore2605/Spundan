"""
Document Search / RAG Tool module.
Connects the Day 7 RAG retrieval pipeline to the Day 8 tool interface.
Separates retrieval and structured data output from presentation formatting.
"""

from typing import Optional, Dict, Any
import logging

from app.tools.base import BaseTool
from app.rag.retriever import RAGRetriever
from app.schemas.tool_schema import ToolResult, ToolDefinition, SearchResponse
from app.config import settings

logger = logging.getLogger(__name__)


# Global / lazily shared retriever instance
_default_retriever: Optional[RAGRetriever] = None


def get_default_retriever() -> RAGRetriever:
    """Retrieves or instantiates the default RAGRetriever singleton."""
    global _default_retriever
    if _default_retriever is None:
        _default_retriever = RAGRetriever(
            documents_dir=settings.documents_dir,
            top_k=settings.top_k_results,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
    return _default_retriever


def document_search(query: str, retriever: Optional[RAGRetriever] = None) -> Dict[str, Any]:
    """
    Standard standalone document search function satisfying Day 8 requirement.
    
    Args:
        query: Natural language query string.
        retriever: Optional RAGRetriever instance.
        
    Returns:
        Dict[str, Any]: Structured search dictionary.
    """
    r = retriever or get_default_retriever()
    response = r.retrieve(query)
    return response.model_dump()


class DocumentSearchTool(BaseTool):
    """Production-grade Document Search Tool implementing BaseTool."""

    def __init__(self, retriever: Optional[RAGRetriever] = None):
        self._retriever = retriever

    @property
    def retriever(self) -> RAGRetriever:
        """Returns the configured or default RAGRetriever."""
        if self._retriever is None:
            self._retriever = get_default_retriever()
        return self._retriever

    @property
    def name(self) -> str:
        return "document_search"

    @property
    def description(self) -> str:
        return (
            "Searches indexed local documents and returns relevant context chunks "
            "with similarity scores and source file citations."
        )

    @property
    def schema(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query or question to retrieve context for, e.g. 'Explain AI'",
                    }
                },
                "required": ["query"],
            },
        )

    def execute(self, input_data: str) -> ToolResult:
        """
        Executes document search retrieval.
        
        Args:
            input_data: String natural language query.
            
        Returns:
            ToolResult containing SearchResponse dict or error message.
        """
        if not input_data or not input_data.strip():
            return ToolResult(
                tool_name=self.name,
                raw_input=input_data,
                success=False,
                error="Error: Query cannot be empty or whitespace.",
            )

        try:
            response = self.retriever.retrieve(input_data)
            return ToolResult(
                tool_name=self.name,
                raw_input=input_data,
                success=True,
                data=response.model_dump(),
            )
        except Exception as e:
            logger.error("Document search error: %s", str(e))
            return ToolResult(
                tool_name=self.name,
                raw_input=input_data,
                success=False,
                error=f"Error during retrieval: {str(e)}",
            )

    def format_output(self, result: ToolResult) -> str:
        """
        Formats retrieved chunks into clean terminal output with source citations.
        Separates presentation logic from pure retrieval logic.
        """
        if not result.success:
            return f"\nDocument Search Error:\n{result.error}\n"

        data = result.data or {}
        results = data.get("results", [])
        query = data.get("query", "")

        if not results:
            return f"\nNo relevant context found in documents for query: '{query}'\n"

        lines = ["\nRetrieved Context:\n"]
        sources = set()

        for idx, item in enumerate(results, start=1):
            content = item.get("content", "").strip()
            source = item.get("source", "unknown")
            score = item.get("score", 0.0)
            sources.add(source)

            lines.append(f"[{idx}] {content}")
            lines.append(f"    Source: {source} (Similarity: {score:.2f})\n")

        lines.append("Sources:")
        for src in sorted(sources):
            lines.append(f"- {src}")
        lines.append("")

        return "\n".join(lines)
