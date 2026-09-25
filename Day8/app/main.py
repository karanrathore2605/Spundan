"""
Day 8 LLM Tools CLI Application.
Provides interactive manual tool execution as well as direct command-line arguments.
Integrates Document Retrieval with Groq LLM Grounded Answer generation.
"""

import sys
import argparse
import logging
from typing import Optional

from app.tools.registry import registry
from app.tools.calculator import CalculatorTool
from app.tools.document_search import DocumentSearchTool
from app.schemas.tool_schema import ToolResult
from app.llm.groq_client import GroqClient
from app.config import settings

logger = logging.getLogger("day8_main")


def print_banner() -> None:
    """Displays the application welcome header."""
    print("=" * 50)
    print("           DAY 8 - LLM TOOLS DEMO")
    print("==================================================")


def print_menu() -> None:
    """Displays the available tools and menu options."""
    print("\nAvailable tools:")
    print("1. Calculator")
    print("2. Document Search & RAG Q&A")
    print("3. Exit")


def display_grounded_answer(query: str, result: ToolResult) -> None:
    """
    Synthesizes and displays the grounded answer using Groq LLM if configured.
    Separates pure retrieval from final LLM generation.
    """
    if not result.success:
        return

    data = result.data or {}
    results = data.get("results", [])
    if not results:
        return

    # Build context string from retrieved chunks
    context_parts = []
    for idx, item in enumerate(results, start=1):
        content = item.get("content", "").strip()
        source = item.get("source", "unknown")
        context_parts.append(f"[{idx}] Source: {source}\n{content}")
    combined_context = "\n\n".join(context_parts)

    print("-" * 50)
    print("Grounded Answer (Groq LLM):")

    groq_client = GroqClient()
    if not groq_client.is_configured:
        print(
            "\n[Note: Groq LLM synthesis is ready. To enable live answer generation,\n"
            " set your GROQ_API_KEY in the .env file. The above retrieved context\n"
            " is the ground truth that will be fed to the model.]"
        )
        print("-" * 50)
        return

    try:
        print("\nSynthesizing answer from retrieved context with Groq...")
        answer = groq_client.generate_grounded_answer(query=query, context=combined_context)
        print(f"\n{answer}\n")
    except Exception as e:
        print(f"\nError generating answer from Groq: {str(e)}\n")
    print("-" * 50)


def run_calculator_flow() -> None:
    """Handles interactive execution of the Calculator tool."""
    tool = registry.get("calculator")
    if not tool:
        print("\nError: Calculator tool is not registered.")
        return

    try:
        expr = input("\nEnter expression: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nInput cancelled.")
        return

    if not expr:
        print("\nError: Expression cannot be empty.")
        return

    result = tool.execute(expr)
    print(tool.format_output(result))


def run_document_search_flow() -> None:
    """Handles interactive execution of Document Search and RAG answer synthesis."""
    tool = registry.get("document_search")
    if not tool:
        print("\nError: Document Search tool is not registered.")
        return

    try:
        query = input("\nEnter query: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nInput cancelled.")
        return

    if not query:
        print("\nError: Query cannot be empty.")
        return

    print("\nSearching documents...")
    result = tool.execute(query)
    print(tool.format_output(result))
    display_grounded_answer(query, result)


def run_interactive() -> None:
    """Main interactive terminal loop for manual tool execution."""
    print_banner()

    while True:
        try:
            print_menu()
            choice = input("\nChoose tool: ").strip()

            if choice in ("1", "calculator"):
                run_calculator_flow()
            elif choice in ("2", "document_search", "search", "rag"):
                run_document_search_flow()
            elif choice in ("3", "exit", "quit", "q"):
                print("\nExiting...")
                break
            else:
                print(f"\nInvalid choice: '{choice}'. Please select 1, 2, or 3.")
        except (KeyboardInterrupt, EOFError):
            print("\n\nExiting...")
            break


def run_direct(tool_name: str, input_value: str) -> None:
    """Executes a single tool directly from command line arguments."""
    normalized_name = "calculator" if tool_name in ("1", "calculator") else "document_search"
    tool = registry.get(normalized_name)
    if not tool:
        print(f"Error: Unknown tool '{tool_name}'. Available: {registry.get_tool_names()}")
        sys.exit(1)

    print(f"Tool selected: {tool.name}")
    print(f"Tool input: {input_value}")
    if normalized_name == "document_search":
        print("Searching documents...")

    result = tool.execute(input_value)
    print(tool.format_output(result))

    if normalized_name == "document_search":
        display_grounded_answer(input_value, result)


def main() -> None:
    """CLI entrypoint supporting both interactive mode and CLI flags."""
    parser = argparse.ArgumentParser(description="Day 8: LLM Tools Manual Execution Demo")
    parser.add_argument("--tool", choices=["calculator", "document_search", "1", "2"], help="Directly choose tool to execute")
    parser.add_argument("--input", dest="tool_input", help="Input expression or query for the tool")

    args = parser.parse_args()

    if args.tool:
        if not args.tool_input:
            print("Error: --input is required when --tool is specified.")
            sys.exit(1)
        run_direct(args.tool, args.tool_input)
    else:
        run_interactive()


if __name__ == "__main__":
    main()
