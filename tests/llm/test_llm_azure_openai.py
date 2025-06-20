"""
Test cases for AzureOpenAILLM class.
"""

import unittest
from unittest.mock import Mock, patch

from elemental_agents.llm.data_model import ModelParameters
from elemental_agents.llm.llm_azure_openai import AzureOpenAILLM


class TestAzureOpenAILLM(unittest.TestCase):
    """
    Test Azure OpenAI LLM implementation.
    """

    @patch("elemental_agents.llm.llm_azure_openai.AzureOpenAI")
    def test_azure_openai_initialization(self, mock_azure_openai: Mock) -> None:
        """
        Test Azure OpenAI LLM initialization.
        """
        llm = AzureOpenAILLM(
            model_name="gpt-4-deployment",
            api_key="test_key",
            azure_endpoint="https://test.openai.azure.com/",
            api_version="2024-02-15-preview",
        )

        self.assertEqual(llm._model, "gpt-4-deployment")
        mock_azure_openai.assert_called_once_with(
            api_key="test_key",
            azure_endpoint="https://test.openai.azure.com/",
            api_version="2024-02-15-preview",
        )

    def test_convert_message_for_azure_openai(self) -> None:
        """
        Test message conversion for Azure OpenAI.
        """
        with patch("elemental_agents.llm.llm_azure_openai.AzureOpenAI"):
            llm = AzureOpenAILLM(model_name="test-deployment")

            message = {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello"},
                    {
                        "type": "image",
                        "base64_data": "test_data",
                        "media_type": "image/png",
                    },
                ],
            }

            result = llm._convert_message_for_azure_openai(message)

            self.assertEqual(result["role"], "user")
            self.assertEqual(len(result["content"]), 2)
            self.assertEqual(result["content"][0]["type"], "text")
            self.assertEqual(result["content"][1]["type"], "image_url")

    @patch("elemental_agents.llm.llm_azure_openai.AzureOpenAI")
    def test_run_non_streaming_with_reasoning(self, mock_azure_openai: Mock) -> None:
        """
        Test non-streaming run with reasoning effort.
        """
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Test response"
        mock_response.usage = Mock()
        mock_response.usage.total_tokens = 150
        mock_client.chat.completions.create.return_value = mock_response
        mock_azure_openai.return_value = mock_client

        params = ModelParameters(reasoning_effort="medium")
        llm = AzureOpenAILLM(model_name="o1-deployment", parameters=params)
        messages = [{"role": "user", "content": "Hello"}]

        result = llm._run_non_streaming(messages, [])

        self.assertEqual(result, "Test response")
        call_args = mock_client.chat.completions.create.call_args
        self.assertEqual(call_args[1]["reasoning_effort"], "medium")


if __name__ == "__main__":
    unittest.main()
