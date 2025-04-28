"""
Safe evaluation of mathematical expressions containing numbers and basic operators.
"""

import ast
import math
import operator
from typing import Any, Callable, Dict, Union


class SafeEvaluator(ast.NodeVisitor):
    """
    Safely evaluate a mathematical expression containing numbers, basic
    operators, and trigonometric functions.
    """

    # Mapping of AST binary operator nodes to functions
    binary_operators: Dict[type, Callable[[Any, Any], Any]] = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.Mod: operator.mod,
        ast.FloorDiv: operator.floordiv,
    }

    # Mapping of AST unary operator nodes to functions
    unary_operators: Dict[type, Callable[[Any], Any]] = {
        ast.USub: operator.neg,
    }

    functions: dict[str, Callable[..., Any]] = {
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "sqrt": math.sqrt,
        "log": math.log,
        "exp": math.exp,
        "asin": math.asin,
        "acos": math.acos,
        "atan": math.atan,
    }

    def visit(self, node: ast.AST) -> Union[float, int, Any]:
        if isinstance(node, ast.Expression):
            return self.visit(node.body)
        if isinstance(node, ast.Constant):  # For Python >= 3.8
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError(f"Unsupported constant: {node.value}")
        if isinstance(node, ast.BinOp):
            left = self.visit(node.left)
            right = self.visit(node.right)
            op_type = type(node.op)
            if op_type in self.binary_operators:
                return self.binary_operators[op_type](left, right)
            raise ValueError(f"Unsupported binary operator: {op_type}")
        if isinstance(node, ast.UnaryOp):
            operand = self.visit(node.operand)
            if isinstance(node.op, tuple(self.unary_operators)):
                # Correctly apply unary operator
                return self.unary_operators[type(node.op)](operand)
            raise ValueError(f"Unsupported unary operator: {type(node.op)}")
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):  # Ensure node.func is ast.Name
                func_name = node.func.id
                if func_name in self.functions:
                    args = [self.visit(arg) for arg in node.args]
                    return self.functions[func_name](*args)
            raise ValueError(f"Unsupported function: {node.func}")

        raise ValueError(f"Unsupported expression: {node}")


def safe_eval(expression: str) -> Union[float, int, Any]:
    """
    Safely evaluate a mathematical expression containing numbers, basic
    operators, and trigonometric functions.
    """
    # Replace ^ with ** for power
    expression = expression.replace("^", "**")

    try:
        parsed_expr = ast.parse(expression, mode="eval")
        evaluator = SafeEvaluator()
        return evaluator.visit(parsed_expr)
    except Exception as e:
        raise ValueError(f"Error evaluating expression: {e}") from e


if __name__ == "__main__":
    NEW_EXPRESSION = "2 + 3 * 4"
    result = safe_eval(NEW_EXPRESSION)
    print(result)  # Output: 14
