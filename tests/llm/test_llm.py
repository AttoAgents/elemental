"""
Test the LLM classes for language models.
"""

import unittest

from elemental_agents.llm.data_model import Message, ModelParameters
from elemental_agents.llm.llm_mock import MockLLM
from elemental_agents.llm.llm_ollama import OllamaLLM
from elemental_agents.llm.llm_openai import OpenAILLM

MODEL_NAME = "llama3:8b"
OPENAI_MOEL = "gpt-4o-mini"


class TestOpenAILLM(unittest.TestCase):
    """
    Test OpenAI models in the LLM class.
    """

    def test_run_messages(self) -> None:
        """
        Test running messages through the OpenAI model class.
        """

        from elemental_agents.utils.config import ConfigModel

        config = ConfigModel()

        model_parameters = ModelParameters(
            temperature=0.0,
            stop=[],
            max_tokens=1000,
        )

        llm = OpenAILLM(
            model_name=OPENAI_MOEL,
            message_stream=False,
            parameters=model_parameters,
            openai_api_key=config.openai_api_key,
        )

        messages = [
            Message(
                role="system",
                content="You are helpful and kind assistant. You always always user's questions and provide helpful answers.",
            ),
            Message(role="user", content="Why is the sky blue?"),
        ]

        response = llm.run(messages)

        self.assertIsNotNone(response)


class TestOllamaLLM(unittest.TestCase):
    """
    Test the OllamaLLM class for language models from Ollama.
    """

    def test_run_with_empty_messages(self) -> None:
        """
        Run OllamaLLM with empty messages.
        """

        llm = OllamaLLM(model_name=MODEL_NAME)
        result = llm.run([])

        self.assertEqual(result, "No messages to process.")

    def test_run_with_user_message(self) -> None:
        """
        Run OllamaLLM with a user message.
        """

        llm = OllamaLLM(model_name=MODEL_NAME)
        messages = [Message(role="user", content="Hello")]

        result = llm.run(messages)
        self.assertIsNotNone(result)

    def test_run_with_system_message(self) -> None:
        """
        Test running OllamaLLM with a system message.
        """
        llm = OllamaLLM(model_name=MODEL_NAME)
        messages = [Message(role="system", content="Welcome")]

        result = llm.run(messages)
        self.assertIsNotNone(result)

    def test_run_with_options(self) -> None:
        """
        Test OllamaLLM with explicit options.
        """

        model_parameters = ModelParameters(
            temperature=0.7,
            stop=["PAUSE"],
            max_tokens=100,
        )

        llm = OllamaLLM(model_name=MODEL_NAME, parameters=model_parameters)
        messages = [Message(role="user", content="Why is the sky blue?")]
        result = llm.run(messages)

        self.assertIsNotNone(result)

    def test_run_with_multiple_messages(self) -> None:
        """
        Test OllamaLLM class with multiple messages.
        """
        llm = OllamaLLM(model_name=MODEL_NAME)
        messages = [
            Message(role="user", content="Hello"),
            Message(role="system", content="How can I help you?"),
        ]
        result = llm.run(messages)
        self.assertIsNotNone(result)


class TestMockLLM(unittest.TestCase):
    """
    Test MockLLM class for testing purposes.
    """

    def test_run_with_empty_messages(self) -> None:
        """
        Initialize MockLLM with empty messages.
        """

        llm = MockLLM()
        result = llm.run([])  # type: ignore

        self.assertEqual(result, "No messages to process.")

    def test_run_with_user_message(self) -> None:
        """
        Test running MockLLM with a user message.
        """

        llm = MockLLM()
        messages = [Message(role="user", content="Hello")]
        result = llm.run(messages)  # type: ignore

        self.assertEqual(result, "Output for user message (Hello)")

    def test_run_with_system_message(self) -> None:
        """
        Test running MockLLM with a system message.
        """

        llm = MockLLM()
        messages = [Message(role="system", content="Welcome")]
        result = llm.run(messages)  # type: ignore

        self.assertEqual(result, "Output for system message (Welcome)")

    def test_run_with_multiple_messages(self) -> None:
        """
        Test running MockLLM with multiple messages.
        """

        llm = MockLLM()
        messages = [
            Message(role="user", content="Hello"),
            Message(role="system", content="How can I help you?"),
        ]
        result = llm.run(messages)  # type: ignore

        self.assertEqual(result, "Output for system message (How can I help you?)")


if __name__ == "__main__":
    unittest.main()
