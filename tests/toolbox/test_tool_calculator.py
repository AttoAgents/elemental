"""
Test calculator tool.
"""

import json
import unittest

from elemental_agents.tools.calculator import Calculator, CalculatorParams


class TestToolCalculator(unittest.TestCase):
    """
    Test calculator tool on simple arithmetic expressions.
    """

    def test_calculator(self) -> None:
        """
        Calculator test: simple addition.
        """

        calculator = Calculator()

        calculator_json = '{"expression": "2 + 3 * 4"}'
        params = json.loads(calculator_json)
        result = calculator.run(CalculatorParams(**params))

        self.assertEqual(result.value, 14)

    def test_calculator_error(self) -> None:
        """
        Calculator test: incomplete expression.
        """

        calculator = Calculator()

        calculator_json = '{"expression": "2 + 3 * "}'
        params = json.loads(calculator_json)
        result = calculator.run(CalculatorParams(**params))

        self.assertEqual(result.value, 0.0)
        self.assertIn("Error", result.status)

    def test_calculator_divide_by_zero(self) -> None:
        """
        Calculator test: division by zero.
        """

        calculator = Calculator()

        calculator_json = '{"expression": "1 / 0"}'
        params = json.loads(calculator_json)
        result = calculator.run(CalculatorParams(**params))

        self.assertEqual(result.value, 0.0)
        self.assertIn("Error", result.status)

    def test_calculator_invalid_expression(self) -> None:
        """
        Calculator test: invalid expression.
        """

        calculator = Calculator()

        calculator_json = '{"expression": "1 / a"}'
        params = json.loads(calculator_json)
        result = calculator.run(CalculatorParams(**params))

        self.assertEqual(result.value, 0.0)
        self.assertIn("Error", result.status)

    def test_calculator_extra_field(self) -> None:
        """
        Calculator test: ignore additional data.
        """

        calculator = Calculator()

        calculator_json = '{"expression": "2 + 3 * 4", "extra": "field"}'
        params = json.loads(calculator_json)
        result = calculator.run(CalculatorParams(**params))

        self.assertEqual(result.value, 14)

    def test_float_expression(self) -> None:
        """
        Calculator test: floating point arithmetic.
        """

        calculator = Calculator()

        calculator_json = '{"expression": "2.5 + 3.5"}'
        params = json.loads(calculator_json)
        result = calculator.run(CalculatorParams(**params))

        self.assertEqual(result.value, 6.0)


if __name__ == "__main__":
    unittest.main()
