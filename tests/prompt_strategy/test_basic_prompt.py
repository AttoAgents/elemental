"""
Test cases for BasicPrompt class.
"""

import unittest

from elemental_agents.core.prompt_strategy.basic_prompt import BasicPrompt
from elemental_agents.core.prompt_strategy.prompt_template import FileTemplate
from elemental_agents.llm.data_model import Message


class TestBasicPrompt(unittest.TestCase):
    """
    Test BasicPrompt class.
    """

    def test_basic_prompt(self) -> None:
        """
        Test basic prompt with a simple template.
        """

        context = {
            "agent_name": "ResearchAgent",
            "agent_persona": "Researcher always following scientific method",
        }

        system_template = FileTemplate(context, "test.template")
        basic_prompt = BasicPrompt(system_template)

        user_input = "This is a test input"

        result = basic_prompt.render(user_input)

        self.assertEqual(
            result[0].content,
            "This is a test template for ResearchAgent. Researcher always following scientific method.",
        )

        self.assertEqual(result[1].content, "This is a test input")

    def test_basic_prompt_with_history(self) -> None:
        """
        Test basic prompt with a simple template and history.
        """

        context = {
            "agent_name": "ResearchAgent",
            "agent_persona": "Researcher always following scientific method",
        }

        system_template = FileTemplate(context, "test.template")
        basic_prompt = BasicPrompt(system_template)

        user_input = "This is a test input"

        history = [
            Message(role="user", content="First user message."),
            Message(
                role="assistant", content="Assistant response after first user message."
            ),
        ]

        result = basic_prompt.render(user_input, history)

        self.assertEqual(
            result[0].content,
            "This is a test template for ResearchAgent. Researcher always following scientific method.",
        )

        self.assertEqual(result[1].content, "First user message.")
        self.assertEqual(
            result[2].content, "Assistant response after first user message."
        )
        self.assertEqual(result[3].content, "This is a test input")


if __name__ == "__main__":
    unittest.main()
