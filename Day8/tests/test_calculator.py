"""
Tests for Safe Calculator Tool.
Verifies arithmetic correctness, security against injection attacks,
and graceful error handling.
"""

import pytest
from app.tools.calculator import safe_calculate, calculator, CalculatorTool
from app.schemas.tool_schema import ToolResult


class TestSafeCalculator:
    """Tests for safe_calculate function and SafeExpressionEvaluator."""

    def test_basic_arithmetic(self):
        assert safe_calculate("5 * 12") == 60
        assert safe_calculate("10 + 5") == 15
        assert safe_calculate("20 - 5") == 15
        assert safe_calculate("20 / 4") == 5
        assert safe_calculate("100 / 4") == 25
        assert safe_calculate("2 ** 5") == 32
        assert safe_calculate("(10 + 5) * 2") == 30

    def test_unary_and_floating_point(self):
        assert safe_calculate("-5 + 10") == 5
        assert safe_calculate("+7 - 2") == 5
        assert safe_calculate("2.5 * 4") == 10
        assert safe_calculate("7 // 2") == 3
        assert safe_calculate("7 % 4") == 3

    def test_division_by_zero(self):
        with pytest.raises(ZeroDivisionError):
            safe_calculate("5 / 0")

        with pytest.raises(ZeroDivisionError):
            safe_calculate("10 // 0")

        with pytest.raises(ZeroDivisionError):
            safe_calculate("10 % 0")

    def test_empty_and_whitespace_input(self):
        with pytest.raises(ValueError, match="empty"):
            safe_calculate("")

        with pytest.raises(ValueError, match="empty"):
            safe_calculate("   ")

    def test_invalid_syntax(self):
        with pytest.raises(ValueError):
            safe_calculate("5 +")

        with pytest.raises(ValueError):
            safe_calculate("++*--")

    def test_security_rejection_of_dangerous_expressions(self):
        """Must reject any malicious code execution, imports, or system calls."""
        dangerous_inputs = [
            "__import__('os').system('ls')",
            "open('documents/sample.txt')",
            "exec('x = 1')",
            "eval('5 + 5')",
            "import os",
            "os.system('dir')",
            "[x for x in range(10)]",
            "lambda x: x + 1",
            "__builtins__",
            "().__class__.__base__.__subclasses__()",
        ]
        for expr in dangerous_inputs:
            with pytest.raises(ValueError):
                safe_calculate(expr)

    def test_dos_protection_large_exponents(self):
        """Guards against exponential DoS attacks like 9999 ** 99999."""
        with pytest.raises(OverflowError):
            safe_calculate("2 ** 10000")


class TestCalculatorStandaloneFunction:
    """Tests for the calculator(expression: str) -> str function."""

    def test_calculator_function_success(self):
        assert calculator("5 * 12") == "60"
        assert calculator("10 + 5") == "15"

    def test_calculator_function_division_by_zero(self):
        output = calculator("5 / 0")
        assert "Division by zero" in output

    def test_calculator_function_dangerous_input(self):
        output = calculator("__import__('os').system('echo hacked')")
        assert "Error:" in output


class TestCalculatorToolClass:
    """Tests for the CalculatorTool class implementing BaseTool."""

    def setup_method(self):
        self.tool = CalculatorTool()

    def test_tool_metadata(self):
        assert self.tool.name == "calculator"
        assert "mathematical" in self.tool.description.lower()
        schema = self.tool.schema
        assert schema.name == "calculator"
        assert "expression" in schema.parameters["properties"]

    def test_tool_execute_success(self):
        result: ToolResult = self.tool.execute("5 * 12")
        assert result.success is True
        assert result.data == 60
        assert result.error is None
        formatted = self.tool.format_output(result)
        assert "60" in formatted

    def test_tool_execute_division_by_zero(self):
        result: ToolResult = self.tool.execute("5 / 0")
        assert result.success is False
        assert "Division by zero" in result.error
        formatted = self.tool.format_output(result)
        assert "Error" in formatted

    def test_tool_execute_empty(self):
        result: ToolResult = self.tool.execute("")
        assert result.success is False
        assert "empty" in result.error.lower()
