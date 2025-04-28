"""
Test currency conversion tool.
"""

import json
import unittest

from elemental_agents.tools.currency_conversion import (
    CurrencyConversion,
    CurrencyConversionParams,
    CurrencyConversionResult,
)
from elemental_agents.utils.exceptions import ToolException


class TestToolCurrency(unittest.TestCase):
    """
    Test currency conversion tool.
    """

    def test_currency_conversion(self) -> None:
        """
        Test currency conversion.
        """

        currency_conversion = CurrencyConversion()

        currency_conversion_json = (
            '{"value": 10, "from_currency": "USD", "to_currency": "EUR"}'
        )
        params = json.loads(currency_conversion_json)
        result = currency_conversion.run(CurrencyConversionParams(**params))

        self.assertGreaterEqual(result.result, 0.0)

    def test_converting_forth_and_back(self) -> None:
        """
        Test converting forth and back.
        """

        currency_conversion = CurrencyConversion()

        param1 = CurrencyConversionParams(
            value=10, from_currency="USD", to_currency="GBP"
        )
        result1 = currency_conversion.run(param1)

        param2 = CurrencyConversionParams(
            value=result1.result, from_currency="GBP", to_currency="USD"
        )
        result2 = currency_conversion.run(param2)

        expected = CurrencyConversionResult(result=10)
        self.assertAlmostEqual(result2.result, expected.result, places=2)

    def test_currency_conversion_error(self) -> None:
        """
        Test currency conversion error.
        """

        currency_conversion = CurrencyConversion()

        currency_conversion_json = (
            '{"value": 1, "from_currency": "USD", "to_currency": "XYZ"}'
        )
        params = json.loads(currency_conversion_json)

        with self.assertRaises(ToolException):
            currency_conversion.run(CurrencyConversionParams(**params))


if __name__ == "__main__":
    unittest.main()
