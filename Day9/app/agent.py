import re
import logging
from typing import Literal

from .tools import calculator_tool
from .rag import rag_tool
from .llm import generate_answer

logger = logging.getLogger(__name__)
Route = Literal["CALCULATOR", "RAG", "LLM"]

_MATH_WORDS = re.compile(r"\b(calculate|compute|solve)\b", re.I)
_MATH_EXPR = re.compile(r"(?<!\w)\d+(?:\s*(?:\+|-|\*|/|%|\*\*)\s*\d+)+(?!\w)")
_KNOWLEDGE_WORDS = re.compile(
    r"\b(what is|what are|who is|explain|define|how does|how do|why is|"
    r"machine learning|deep learning|transformer|transformers|rag|"
    r"embedding|embeddings|backpropagation|attention|neural network)\b",
    re.I,
)

def _extract_math_expression(text: str) -> str:
    match = _MATH_EXPR.search(text)
    if match:
        return match.group(0)
    # For simple "calculate 25 * 8" style inputs, remove common words.
    cleaned = re.sub(r"(?i)\b(what is|calculate|compute|solve)\b", "", text)
    cleaned = cleaned.replace("?", "").strip()
    if re.fullmatch(r"[0-9\s+\-*/%().]+", cleaned):
        return cleaned
    return ""

def detect_route(question: str) -> tuple[Route, str | None]:
    math_expr = _extract_math_expression(question)
    if math_expr or _MATH_WORDS.search(question):
        if math_expr:
            return "CALCULATOR", math_expr
    if _KNOWLEDGE_WORDS.search(question):
        return "RAG", None
    return "LLM", None

class RuleBasedAgent:
    def run(self, question: str) -> dict:
        question = question.strip()
        if not question:
            return {"success": False, "tool": "NONE", "answer": "Please enter a question.", "sources": []}

        route, expression = detect_route(question)
        logger.info("Route selected: %s", route)

        try:
            if route == "CALCULATOR":
                if not expression:
                    return {
                        "success": False,
                        "tool": "CALCULATOR",
                        "answer": "I detected a calculation, but could not safely extract the mathematical expression.",
                        "sources": [],
                    }
                return {
                    "success": True,
                    "tool": "CALCULATOR",
                    "answer": calculator_tool(expression),
                    "sources": [],
                }

            if route == "RAG":
                result = rag_tool(question)
                # If no documents are available, use a controlled direct-LLM fallback.
                if not result["success"]:
                    logger.warning("RAG had no usable context; falling back to direct LLM.")
                    answer = generate_answer(question)
                    return {
                        "success": True,
                        "tool": "LLM_FALLBACK",
                        "answer": answer,
                        "sources": [],
                    }
                result["tool"] = "RAG"
                return result

            return {
                "success": True,
                "tool": "LLM",
                "answer": generate_answer(question),
                "sources": [],
            }
        except Exception as exc:
            logger.exception("Agent execution failed")
            return {
                "success": False,
                "tool": route,
                "answer": f"Request failed: {exc}",
                "sources": [],
            }
