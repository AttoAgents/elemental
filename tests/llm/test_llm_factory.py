"""
Test cases for LLMFactory class.
"""

import unittest

from elemental_agents.llm.data_model import Message
from elemental_agents.llm.llm import LLM
from elemental_agents.llm.llm_factory import LLMFactory


class TestLLMFactory(unittest.TestCase):
    """
    Test LLM factory class.
    """

    def test_llm_factory(self) -> None:
        """
        Test default LLM factory.
        """

        llm_factory = LLMFactory()
        llm = llm_factory.create()

        msg = Message(role="user", content="Hello, how are you?")
        result = llm.run([msg])

        self.assertIsInstance(llm, LLM)
        self.assertIsNotNone(result)

    def test_llm_factory_ollama(self) -> None:
        """
        Test Ollama model from LLM factory.
        """

        llm_factory = LLMFactory()
        llm = llm_factory.create("ollama|gemma3")

        msg = Message(role="user", content="Hello, how are you?")
        result1 = llm.run([msg])

        self.assertIsInstance(llm, LLM)
        self.assertIsNotNone(result1)

        msg2 = Message(role="user", content="What is the weather like today?")
        result2 = llm.run([msg2], "STOP")

        self.assertIsNotNone(result2)

    def test_llm_factory_openai(self) -> None:
        """
        Test OpenAI model from LLM factory.
        """

        llm_factory = LLMFactory()
        llm = llm_factory.create("openai|gpt-4.1-mini")

        msg = Message(role="user", content="Hello, how are you?")
        result1 = llm.run([msg])

        self.assertIsInstance(llm, LLM)
        self.assertIsNotNone(result1)

        msg2 = Message(role="user", content="What is the weather like today?")
        result2 = llm.run([msg2], "STOP")

        self.assertIsNotNone(result2)

    def test_llm_factory_mock(self) -> None:
        """
        Test mock model from LLM factory.
        """

        llm_factory = LLMFactory()
        llm = llm_factory.create("mock")

        msg = Message(role="user", content="Hello, how are you?")
        result1 = llm.run([msg])

        self.assertIsInstance(llm, LLM)
        self.assertIsNotNone(result1)

    def test_engine_and_model_name_ollama(self) -> None:
        """
        Test Ollama engine and model from LLM factory.
        """

        llm_factory = LLMFactory()
        llm = llm_factory.create("ollama|llama3.2:latest")

        msg = Message(role="user", content="Hello, how are you?")
        result1 = llm.run([msg])

        self.assertIsInstance(llm, LLM)
        self.assertIsNotNone(result1)

    def test_engine_and_model_name_openai(self) -> None:
        """
        Test OpenAI engine and model from LLM factory.
        """

        llm_factory = LLMFactory()
        llm = llm_factory.create("openai|gpt-4o-mini")

        msg = Message(role="user", content="Hello, how are you?")
        result1 = llm.run([msg])

        self.assertIsInstance(llm, LLM)
        self.assertIsNotNone(result1)


if __name__ == "__main__":
    unittest.main()
