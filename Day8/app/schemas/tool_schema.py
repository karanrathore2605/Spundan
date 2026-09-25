"""
Tool schema definitions and data contracts.

Defines:
- ToolResult: Structured output of any tool execution.
- ToolDefinition: Metadata and input schema for tools (usable by LLMs).
- SearchResultChunk and SearchResponse: Structured output for document search.
"""

from typing import Any, Optional, List, Dict
from pydantic import BaseModel, Field


class SearchResultChunk(BaseModel):
    """Represents a single retrieved chunk with source metadata and similarity score."""
    content: str = Field(..., description="The textual chunk content")
    source: str = Field(..., description="The source file or origin of the chunk")
    score: float = Field(..., description="Cosine similarity score (0.0 to 1.0)")


class SearchResponse(BaseModel):
    """Structured response from the document retrieval tool."""
    query: str = Field(..., description="The original query used for search")
    results: List[SearchResultChunk] = Field(default_factory=list, description="List of matched chunks")
    total_found: int = Field(0, description="Total chunks matched")


class ToolResult(BaseModel):
    """
    Standardized result contract for all tool executions.
    Decouples raw execution result from presentation layers.
    """
    tool_name: str = Field(..., description="Name of the tool that executed")
    raw_input: str = Field(..., description="Input string passed to the tool")
    success: bool = Field(..., description="Whether the tool execution succeeded")
    data: Optional[Any] = Field(None, description="Structured execution payload on success")
    error: Optional[str] = Field(None, description="Human-readable error message on failure")


class ToolDefinition(BaseModel):
    """
    Metadata describing a tool to both humans and LLMs.
    Compatible with Groq / OpenAI function calling schema specifications.
    """
    name: str = Field(..., description="Unique identifier for the tool")
    description: str = Field(..., description="Explanation of what the tool does and when to invoke it")
    parameters: Dict[str, Any] = Field(..., description="JSON Schema object describing the arguments")

    def to_groq_format(self) -> Dict[str, Any]:
        """Converts this definition into Groq/OpenAI function calling tool format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
            }
        }
