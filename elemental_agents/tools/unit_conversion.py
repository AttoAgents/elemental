"""
Unit conversion tool for converting between different units of measurement in
metric and imperial systems.
"""

from loguru import logger
from pydantic import Field

from elemental_agents.core.toolbox.tool import Tool, ToolParameters, ToolResult
from elemental_agents.utils.exceptions import ToolException


class UnitConversionParams(ToolParameters):
    """
    Parameters for the UnitConversion tool

    :param value: The value to convert
    :param from_unit: The unit to convert from
    :param to_unit: The unit to convert to
    """

    value: float = Field(..., description="The value to convert")
    from_unit: str = Field(..., description="The unit to convert from")
    to_unit: str = Field(..., description="The unit to convert to")


class UnitConversionResult(ToolResult):
    """
    Result for the UnitConversion tool

    :param result: The converted value
    """

    result: float = Field(..., description="The converted value")

    def __str__(self) -> str:
        return f"{self.result}"


class UnitConversion(Tool):
    """
    Tool for converting values between metric and imperial units of measurement.
    """

    name = "UnitConversion"
    description = "Converts values between metric and imperial units of measurement."

    def run(self, parameters: UnitConversionParams) -> UnitConversionResult:  # type: ignore
        """
        Run the unit conversion tool with the given parameters.
        """

        # Length Conversion factors
        length_conversion = {
            "m_to_ft": 3.28084,
            "ft_to_m": 0.3048,
            "km_to_mi": 0.621371,
            "mi_to_km": 1.60934,
            "cm_to_in": 0.393701,
            "in_to_cm": 2.54,
        }

        # Weight (Mass) Conversion factors
        weight_conversion = {
            "kg_to_lb": 2.20462,
            "lb_to_kg": 0.453592,
            "g_to_oz": 0.035274,
            "oz_to_g": 28.3495,
        }

        # Temperature conversion (special case)
        def c_to_f(celsius: float) -> float:
            return (celsius * 9 / 5) + 32

        def f_to_c(fahrenheit: float) -> float:
            return (fahrenheit - 32) * 5 / 9

        from_unit = parameters.from_unit
        to_unit = parameters.to_unit
        value = parameters.value

        final_value = None

        # Check for length conversions
        if from_unit == "m" and to_unit == "ft":
            final_value = value * length_conversion["m_to_ft"]
        elif from_unit == "ft" and to_unit == "m":
            final_value = value * length_conversion["ft_to_m"]
        elif from_unit == "km" and to_unit == "mi":
            final_value = value * length_conversion["km_to_mi"]
        elif from_unit == "mi" and to_unit == "km":
            final_value = value * length_conversion["mi_to_km"]
        elif from_unit == "cm" and to_unit == "in":
            final_value = value * length_conversion["cm_to_in"]
        elif from_unit == "in" and to_unit == "cm":
            final_value = value * length_conversion["in_to_cm"]

        # Check for weight conversions
        elif from_unit == "kg" and to_unit == "lb":
            final_value = value * weight_conversion["kg_to_lb"]
        elif from_unit == "lb" and to_unit == "kg":
            final_value = value * weight_conversion["lb_to_kg"]
        elif from_unit == "g" and to_unit == "oz":
            final_value = value * weight_conversion["g_to_oz"]
        elif from_unit == "oz" and to_unit == "g":
            final_value = value * weight_conversion["oz_to_g"]

        # Check for temperature conversions
        elif from_unit == "C" and to_unit == "F":
            final_value = c_to_f(value)
        elif from_unit == "F" and to_unit == "C":
            final_value = f_to_c(value)

        # If units are not recognized
        else:
            logger.error("Conversion not supported or units are invalid.")
            raise ToolException("Conversion not supported or units are invalid.")

        result = UnitConversionResult(result=final_value)
        return result
