"""
Tool Registry module for Day 8 LLM Tools.
Provides centralized registration, lookup, and schema exposure for all tools.
Easily extensible to future tools (e.g., weather, web_search, database_search).
"""

from typing import Dict, List, Optional, Callable, Any
import logging

from app.tools.base import BaseTool
from app.tools.calculator import CalculatorTool, calculator
from app.tools.document_search import DocumentSearchTool, document_search
from app.schemas.tool_schema import ToolResult

logger = logging.getLogger(__name__)


class ToolRegistry:
    """
    Registry managing available tools in the application.
    Supports dynamic registration and retrieval of tool instances.
    """

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """Registers a new tool instance."""
        if tool.name in self._tools:
            logger.warning("Overwriting already registered tool: %s", tool.name)
        self._tools[tool.name] = tool
        logger.debug("Registered tool: %s", tool.name)

    def get(self, name: str) -> Optional[BaseTool]:
        """Retrieves a registered tool by its name."""
        return self._tools.get(name)

    def list_tools(self) -> List[BaseTool]:
        """Returns a list of all registered tool instances."""
        return list(self._tools.values())

    def get_tool_names(self) -> List[str]:
        """Returns a list of registered tool names."""
        return list(self._tools.keys())

    def get_groq_schemas(self) -> List[Dict[str, Any]]:
        """Returns Groq/OpenAI compatible function calling schemas for all registered tools."""
        return [tool.schema.to_groq_format() for tool in self._tools.values()]

    def execute(self, tool_name: str, input_data: str) -> ToolResult:
        """
        Executes a registered tool by name with safety checks.
        
        Args:
            tool_name: Name of tool to execute.
            input_data: String input argument.
            
        Returns:
            ToolResult: Execution outcome.
        """
        tool = self.get(tool_name)
        if not tool:
            return ToolResult(
                tool_name=tool_name,
                raw_input=input_data,
                success=False,
                error=f"Error: Tool '{tool_name}' is not registered in the system.",
            )
        return tool.execute(input_data)


# Instantiate pre-configured default registry with Day 8 tools
registry = ToolRegistry()
calculator_tool = CalculatorTool()
document_search_tool = DocumentSearchTool()

registry.register(calculator_tool)
registry.register(document_search_tool)

# Simple dictionary mapping complying with requirement 7
TOOLS: Dict[str, Callable[[str], Any]] = {
    "calculator": calculator,
    "document_search": document_search,
}

TOOL_OBJECTS: Dict[str, BaseTool] = {
    "calculator": calculator_tool,
    "document_search": document_search_tool,
}
