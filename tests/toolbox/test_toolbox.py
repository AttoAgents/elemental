"""
Test toolbox functionality.
"""

import unittest

from elemental_agents.core.toolbox.toolbox import ToolBox
from elemental_agents.tools.calculator import Calculator, CalculatorParams
from elemental_agents.tools.list_files import ListFiles, ListFilesParams


class TestToolbox(unittest.TestCase):
    """
    Test toolbox functionality with adding tools and calling them.
    """

    def test_toolbox(self) -> None:
        """
        Test simple creation of toolbox and calling a tool.
        """

        toolbox = ToolBox()

        toolbox.register_tool("Calculator", Calculator, CalculatorParams)  # type: ignore

        calculator_json = '{"expression": "2 + 3 * 4"}'
        result = toolbox.call_tool("Calculator", calculator_json)

        self.assertEqual(result.value, 14)

    def test_register_tool_by_name(self) -> None:
        """
        Test registering a tool by name.
        """

        toolbox = ToolBox()

        toolbox.register_tool_by_name("Calculator")

        calculator_json = '{"expression": "2 + 3 * 4"}'
        result = toolbox.call_tool("Calculator", calculator_json)

        self.assertEqual(result.value, 14)

    def test_string_representation(self) -> None:
        """
        Return string representation of the result.
        """

        toolbox = ToolBox()

        toolbox.register_tool_by_name("Calculator")

        calculator_json = '{"expression": "2 + 3 * 4"}'
        result = toolbox.call_tool("Calculator", calculator_json)

        expected_str = "Result: 14.0, Status: OK"

        self.assertEqual(str(result), expected_str)

    def test_toolbox_description(self) -> None:
        """
        Test toolbox self-description.
        """

        toolbox = ToolBox()

        toolbox.register_tool("Calculator", Calculator, CalculatorParams)  # type: ignore
        toolbox.register_tool("ListFiles", ListFiles, ListFilesParams)  # type: ignore

        description = toolbox.describe()

        self.assertIn("Calculator", description)
        self.assertIn("ListFiles", description)


if __name__ == "__main__":
    unittest.main()
