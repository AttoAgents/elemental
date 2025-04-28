"""
Test ReactPrompt class.
"""

import unittest

from elemental_agents.core.prompt_strategy.prompt_template import FileTemplate
from elemental_agents.core.prompt_strategy.react_prompt import ReactPrompt
from elemental_agents.llm.data_model import Message


class TestReactPrompt(unittest.TestCase):
    """
    Test ReactPrompt class.
    """

    def test_react_prompt(self) -> None:
        """
        Test ReactPrompt with a simple template.
        """

        context = {
            "agent_name": "ResearchAgent",
            "agent_persona": "Researcher always following scientific method",
        }

        tool_dictionary = {"tool1": "Tool 1 description", "tool2": "Tool 2 description"}

        system_template = FileTemplate(context, "ReAct.template")
        react_prompt = ReactPrompt(system_template, tool_dictionary)

        user_input = "This is a test input"

        result = react_prompt.render(user_input)

        template_snippet = "You are an assistant that responds in the structured format with <thought>, <action>, <PAUSE>, and <observation>."

        self.assertEqual(result[0].content[:30], template_snippet[:30])
        self.assertEqual(result[1].content, "This is a test input")

    def test_react_prompt_with_history(self) -> None:
        """
        Test ReactPrompt with a simple template and history.
        """

        context = {
            "agent_name": "ResearchAgent",
            "agent_persona": "Researcher always following scientific method",
        }

        tool_dictionary = {"tool1": "Tool 1 description", "tool2": "Tool 2 description"}

        system_template = FileTemplate(context, "ReAct.template")
        react_prompt = ReactPrompt(system_template, tool_dictionary)

        user_input = "This is a test input"

        history = [
            Message(role="user", content="First user message."),
            Message(
                role="assistant", content="Assistant response after first user message."
            ),
        ]

        result = react_prompt.render(user_input, history)

        template_snippet = "You are an assistant that responds in the structured format with <thought>, <action>, <PAUSE>, and <observation>."

        self.assertEqual(result[0].content[:30], template_snippet[:30])
        self.assertEqual(result[1].content, "First user message.")
        self.assertEqual(
            result[2].content, "Assistant response after first user message."
        )
        self.assertEqual(result[3].content, "This is a test input")

        self.assertEqual(result[0].role, "system")
        self.assertEqual(result[1].role, "user")
        self.assertEqual(result[2].role, "assistant")
        self.assertEqual(result[3].role, "user")

    def test_react_system_prompt(self) -> None:
        """
        Test ReactPrompt with a simple template.
        """

        context = {
            "agent_name": "ResearchAgent",
            "agent_persona": "Researcher always following scientific method",
        }

        tool_dictionary = {"tool1": "Tool 1 description", "tool2": "Tool 2 description"}

        system_template = FileTemplate(context, "ReAct.template")
        react_prompt = ReactPrompt(system_template, tool_dictionary)

        system_prompt = react_prompt.render_system()

        self.assertIsNotNone(system_prompt)


if __name__ == "__main__":
    unittest.main()
