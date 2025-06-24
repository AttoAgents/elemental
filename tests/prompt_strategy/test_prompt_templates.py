"""
Test cases for prompt templates.
"""

import unittest

from elemental_agents.core.prompt_strategy.prompt_template import (
    FileTemplate,
    StringTemplate,
)


class TestPromptTemplate(unittest.TestCase):
    """
    Test prompt templates.
    """

    def test_file_template(self) -> None:
        """
        Test file template rendering.
        """

        context = {
            "agent_name": "ResearchAgent",
            "agent_persona": "Researcher always following scientific method",
        }

        final_prompt = "This is a test template for ResearchAgent. Researcher always following scientific method."

        file_template = FileTemplate(context, "test.template")

        self.assertEqual(file_template.render(), final_prompt)

    def test_string_template(self) -> None:
        """
        Test string template rendering.
        """

        context = {
            "agent_name": "ResearchAgent",
            "agent_persona": "Researcher always following scientific method",
        }

        final_prompt = "This is a test template for ResearchAgent. Researcher always following scientific method."
        template_string = (
            "This is a test template for {{agent_name}}. {{agent_persona}}."
        )

        string_template = StringTemplate(context, template_string)

        self.assertEqual(string_template.render(), final_prompt)


if __name__ == "__main__":
    unittest.main()
