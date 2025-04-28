"""
Calculator tool for evaluating mathematical expressions.
"""

from loguru import logger
from pydantic import Field

from elemental_agents.core.toolbox.tool import Tool, ToolParameters, ToolResult
from elemental_agents.utils.safe_eval import safe_eval


# Define specific Pydantic model for Calculator input
class CalculatorParams(ToolParameters):
    """
    Parameters for the Calculator tool

    :param expression: Mathematical expression to evaluate
    """

    expression: str = Field(..., description="Mathematical expression to evaluate")


class CalculatorResult(ToolResult):
    """
    Result for the Calculator tool

    :param value: The result of the mathematical expression as a float
    """

    value: float = Field(
        ..., description="The result of the mathematical expression as a float"
    )
    status: str = Field(default="OK", description="The status of the calculation")

    def __str__(self) -> str:
        return f"Result: {self.value}, Status: {self.status}"


# Calculator tool with class-level name and description
class Calculator(Tool):
    """
    Simple calculator tool that evaluates a mathematical expression and returns
    the result as a float.
    """

    name = "Calculator"
    description = (
        "Performs basic arithmetic. Only supports numbers, operators and functions."
    )

    def run(self, parameters: CalculatorParams) -> CalculatorResult:  # type: ignore
        """
        Run the calculator tool with the given parameters.

        :param parameters: The parameters for the calculator tool.
        :return: The result of the mathematical expression as a float.
        """
        try:
            # Evaluate the mathematical expression
            logger.info(f"Evaluating expression: {parameters.expression}")
            value = safe_eval(parameters.expression)
            final_value = float(value)

            result = CalculatorResult(value=final_value, status="OK")
            return result

        except (ValueError, SyntaxError, TypeError) as e:
            logger.error(f"Error evaluating expression: {e}")
            result = CalculatorResult(value=0.0, status=f"Error: {e}")
            return result
