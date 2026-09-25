"""
Safe Calculator Tool module.
Evaluates mathematical expressions safely using Python's Abstract Syntax Tree (ast).
Strictly prevents code injection, arbitrary execution, and resource exhaustion.
"""

import ast
import operator
from typing import Union, Dict, Any, Type
import logging

from app.tools.base import BaseTool
from app.schemas.tool_schema import ToolResult, ToolDefinition

logger = logging.getLogger(__name__)


# Supported binary operators mapped to standard operator functions
SAFE_OPERATORS: Dict[Type[ast.AST], Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

# Supported unary operators
SAFE_UNARY_OPERATORS: Dict[Type[ast.AST], Any] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

# Security limits to prevent Denial-of-Service via massive calculations
MAX_EXPRESSION_LENGTH = 200
MAX_EXPONENT_VALUE = 1000
MAX_BASE_VALUE = 1000000


class SafeExpressionEvaluator(ast.NodeVisitor):
    """
    Safely traverses and evaluates an AST for simple arithmetic expressions.
    Rejects any AST nodes corresponding to function calls, imports, attributes, or variable names.
    """

    def visit(self, node: ast.AST) -> Union[int, float]:
        """Dispatches node to specific visitor or raises error for disallowed nodes."""
        method = "visit_" + node.__class__.__name__
        visitor = getattr(self, method, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node: ast.AST) -> None:
        """Deny any node type not explicitly supported."""
        raise ValueError(f"Dangerous or unsupported syntax detected: '{node.__class__.__name__}'")

    def visit_Expression(self, node: ast.Expression) -> Union[int, float]:
        """Root node for an expression."""
        return self.visit(node.body)

    def visit_Constant(self, node: ast.Constant) -> Union[int, float]:
        """Handles literals (numbers in Python 3.8+)."""
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Unsupported constant type: {type(node.value).__name__}. Only numbers are allowed.")

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Union[int, float]:
        """Handles unary operations like +5 or -10."""
        op_type = type(node.op)
        if op_type not in SAFE_UNARY_OPERATORS:
            raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
        
        operand = self.visit(node.operand)
        return SAFE_UNARY_OPERATORS[op_type](operand)

    def visit_BinOp(self, node: ast.BinOp) -> Union[int, float]:
        """Handles binary operations: +, -, *, /, //, %, **."""
        op_type = type(node.op)
        if op_type not in SAFE_OPERATORS:
            raise ValueError(f"Unsupported binary operator: {op_type.__name__}")

        left = self.visit(node.left)
        right = self.visit(node.right)

        # Guard against division by zero
        if op_type in (ast.Div, ast.FloorDiv, ast.Mod):
            if right == 0:
                raise ZeroDivisionError("Division by zero")

        # Guard against power DoS (e.g. 999999 ** 999999)
        if op_type is ast.Pow:
            if abs(right) > MAX_EXPONENT_VALUE or abs(left) > MAX_BASE_VALUE:
                raise OverflowError("Exponentiation exceeds safety limits.")

        result = SAFE_OPERATORS[op_type](left, right)
        
        # Round float if cleanly an integer
        if isinstance(result, float) and result.is_integer():
            return int(result)
        return result


def safe_calculate(expression: str) -> Union[int, float]:
    """
    Parses and evaluates an arithmetic expression safely without eval().
    
    Args:
        expression: Mathematical expression string (e.g., '5 * 12', '(10 + 5) * 2').
        
    Returns:
        int or float: Computed result.
        
    Raises:
        ValueError, ZeroDivisionError, OverflowError, SyntaxError.
    """
    if not expression or not expression.strip():
        raise ValueError("Expression cannot be empty.")

    clean_expr = expression.strip()
    if len(clean_expr) > MAX_EXPRESSION_LENGTH:
        raise ValueError(f"Expression too long (max {MAX_EXPRESSION_LENGTH} characters).")

    # Parse into AST
    try:
        parsed_ast = ast.parse(clean_expr, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"Invalid mathematical syntax: '{clean_expr}'") from e

    evaluator = SafeExpressionEvaluator()
    return evaluator.visit(parsed_ast)


def calculator(expression: str) -> str:
    """
    Standard standalone calculator function satisfying Day 8 requirement.
    
    Args:
        expression: Mathematical expression string.
        
    Returns:
        str: Output string (computed value or error message).
    """
    try:
        result = safe_calculate(expression)
        return str(result)
    except ZeroDivisionError:
        return "Error: Division by zero"
    except Exception as e:
        return f"Error: {str(e)}"


class CalculatorTool(BaseTool):
    """Production-grade Calculator Tool implementing BaseTool."""

    @property
    def name(self) -> str:
        return "calculator"

    @property
    def description(self) -> str:
        return (
            "Performs safe mathematical calculations. Supports arithmetic operations "
            "(+, -, *, /, //, %, **) and parentheses. Rejects unsafe code execution."
        )

    @property
    def schema(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The mathematical expression to evaluate, e.g. '5 * 12' or '(10 + 5) * 2'",
                    }
                },
                "required": ["expression"],
            },
        )

    def execute(self, input_data: str) -> ToolResult:
        """
        Executes the calculator tool with safe evaluation.
        
        Args:
            input_data: String mathematical expression.
            
        Returns:
            ToolResult containing computation result or error message.
        """
        if not input_data or not input_data.strip():
            return ToolResult(
                tool_name=self.name,
                raw_input=input_data,
                success=False,
                error="Error: Expression cannot be empty.",
            )

        try:
            val = safe_calculate(input_data)
            return ToolResult(
                tool_name=self.name,
                raw_input=input_data,
                success=True,
                data=val,
            )
        except ZeroDivisionError:
            return ToolResult(
                tool_name=self.name,
                raw_input=input_data,
                success=False,
                error="Error: Division by zero",
            )
        except Exception as e:
            logger.debug("Calculator execution error: %s", str(e))
            return ToolResult(
                tool_name=self.name,
                raw_input=input_data,
                success=False,
                error=f"Error: {str(e)}",
            )

    def format_output(self, result: ToolResult) -> str:
        """Formats the computation result for terminal display."""
        if result.success:
            return f"\nCalculator Result:\n{result.data}\n"
        return f"\nCalculator Error:\n{result.error}\n"
