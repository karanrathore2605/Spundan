"""
Base Tool abstraction for Day 8 LLM Tools.
Defines the standard contract for tool execution, metadata, and presentation formatting.
"""

from abc import ABC, abstractmethod
from typing import Any
from app.schemas.tool_schema import ToolResult, ToolDefinition


class BaseTool(ABC):
    """
    Abstract base class representing an executable tool for an LLM system.
    
    Attributes:
        name (str): Unique identifier of the tool.
        description (str): Explains what the tool does (used by LLMs for tool selection).
        schema (ToolDefinition): Structured JSON schema describing tool parameters.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for the tool."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Clear description of the tool's purpose and usage."""
        pass

    @property
    @abstractmethod
    def schema(self) -> ToolDefinition:
        """Structured parameter schema for LLM function calling."""
        pass

    @abstractmethod
    def execute(self, input_data: str) -> ToolResult:
        """
        Executes the tool with the given input string.
        
        Args:
            input_data: String input passed from user or LLM tool-calling agent.
            
        Returns:
            ToolResult: Standardized result containing status, payload, or error.
        """
        pass

    @abstractmethod
    def format_output(self, result: ToolResult) -> str:
        """
        Formats a ToolResult for user-facing terminal presentation.
        Keeps execution/retrieval logic separate from presentation logic.
        
        Args:
            result: The ToolResult returned by execute().
            
        Returns:
            str: Human-readable formatted string.
        """
        pass
