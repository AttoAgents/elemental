"""
Test extraction base on XML-like tags.
"""

import unittest

from elemental_agents.utils.utils import extract_tag_content


class TestExtractTagContent(unittest.TestCase):
    """
    Test extracting content between XML-like tags in a text string.
    """

    def test_extract_tag_content(self) -> None:
        """
        Test extracting content between XML-like tags in a text string.
        """

        text = """
        <XML>
            <JSON>{"key": "value"}</JSON>
            <JSON>{"key": "value"}</JSON>
            <JSON>{"key": "value"}</JSON>
        </XML>
        """

        tag = "JSON"

        content = extract_tag_content(text, tag)

        self.assertEqual(
            content, ['{"key": "value"}', '{"key": "value"}', '{"key": "value"}']
        )


if __name__ == "__main__":
    unittest.main()
