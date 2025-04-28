"""
Test llm.llama_cpp inference in the llm module.
"""

import unittest

from loguru import logger

from elemental_agents.llm.data_model import Message, ModelParameters
from elemental_agents.llm.llm_llama_cpp import LlamaCppLLM


class TestLlamaCppLLM(unittest.TestCase):
    """
    Test the Llama_cpp inference.
    """

    def test_run(self) -> None:
        """
        Test the run method of the LlamaCppLLM class.
        """
        # Create the LlamaCppLLM object
        parameters = ModelParameters(temperature=0.0)

        llm = LlamaCppLLM(
            model_name="models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
            message_stream=False,
            parameters=parameters,
        )

        # Create the messages
        messages = [Message(content="Why is the sky blue?", role="user")]

        # Run the model
        response = llm.run(messages=messages)

        logger.info(f"Response: {response}")

        # Check the response
        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 0)


if __name__ == "__main__":
    unittest.main()
