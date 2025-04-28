"""
Test unit conversion tool.
"""

import json
import unittest

from elemental_agents.tools.unit_conversion import (
    UnitConversion,
    UnitConversionParams,
    UnitConversionResult,
)
from elemental_agents.utils.exceptions import ToolException


class TestToolUnitConversion(unittest.TestCase):
    """
    Test conversion tool for values between metric and imperial units of measurement.
    """

    def test_unit_conversion_simple(self) -> None:
        """
        Test unit conversion: meters to centimeters.
        """

        unit_conversion = UnitConversion()

        unit_conversion_json = '{"value": 1, "from_unit": "cm", "to_unit": "in"}'
        params = json.loads(unit_conversion_json)
        result = unit_conversion.run(UnitConversionParams(**params))

        expected = UnitConversionResult(result=0.393701)

        self.assertAlmostEqual(result.result, expected.result, places=4)
        self.assertAlmostEqual(result.result, 0.393701, places=4)

    def test_unit_conversion_distance(self) -> None:
        """
        Test unit conversion: meters to centimeters.
        """

        unit_conversion = UnitConversion()

        unit_conversion_json = '{"value": 1.5, "from_unit": "km", "to_unit": "mi"}'
        params = json.loads(unit_conversion_json)
        result = unit_conversion.run(UnitConversionParams(**params))

        expected = UnitConversionResult(result=0.932057)

        self.assertAlmostEqual(result.result, expected.result, places=4)
        self.assertAlmostEqual(result.result, 0.932057, places=4)

    def test_unit_conversion_temperature(self) -> None:
        """
        Test unit conversion: meters to centimeters.
        """

        unit_conversion = UnitConversion()

        unit_conversion_json = '{"value": 32, "from_unit": "F", "to_unit": "C"}'
        params = json.loads(unit_conversion_json)
        result = unit_conversion.run(UnitConversionParams(**params))

        expected = UnitConversionResult(result=0)

        self.assertAlmostEqual(result.result, expected.result, places=4)
        self.assertAlmostEqual(result.result, 0.00000)

    def test_unit_conversion_weight(self) -> None:
        """
        Test unit conversion: meters to centimeters.
        """

        unit_conversion = UnitConversion()

        unit_conversion_json = '{"value": 2, "from_unit": "lb", "to_unit": "kg"}'
        params = json.loads(unit_conversion_json)
        result = unit_conversion.run(UnitConversionParams(**params))

        expected = UnitConversionResult(result=0.907185)

        self.assertAlmostEqual(result.result, expected.result, places=4)
        self.assertAlmostEqual(result.result, 0.907185, places=4)

    def test_unit_conversion_error(self) -> None:
        """
        Test unit conversion: invalid conversion.
        """

        unit_conversion = UnitConversion()

        unit_conversion_json = '{"value": 1, "from_unit": "m", "to_unit": "c"}'
        params = json.loads(unit_conversion_json)

        with self.assertRaises(ToolException):
            unit_conversion.run(UnitConversionParams(**params))

            unit_conversion.run(UnitConversionParams(**params))


if __name__ == "__main__":
    unittest.main()
