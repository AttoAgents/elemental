"""
Test cases for BedrockAnthropicLLM class.
"""

import json
import unittest
from unittest.mock import Mock, patch

from elemental_agents.llm.data_model import ModelParameters
from elemental_agents.llm.llm_bedrock_anthropic import BedrockAnthropicLLM


class TestBedrockAnthropicLLM(unittest.TestCase):
    """
    Test Bedrock Anthropic LLM implementation.
    """

    @patch("boto3.Session")
    def test_bedrock_anthropic_initialization(self, mock_session: Mock) -> None:
        """
        Test Bedrock Anthropic LLM initialization.
        """
        mock_session_instance = Mock()

        mock_client = Mock()
        mock_session_instance.client.return_value = mock_client
        mock_session.return_value = mock_session_instance

        llm = BedrockAnthropicLLM(
            model_name="anthropic.claude-3-sonnet-20240229-v1:0",
            aws_access_key_id="test_access_key",
            aws_secret_access_key="test_secret_key",
            aws_session_token="test_token",
            region_name="us-west-2",
        )

        self.assertEqual(llm._model, "anthropic.claude-3-sonnet-20240229-v1:0")
        self.assertEqual(llm._region_name, "us-west-2")

        mock_session.assert_called_once_with(
            aws_access_key_id="test_access_key",
            aws_secret_access_key="test_secret_key",
            aws_session_token="test_token",
            region_name="us-west-2",
        )
        mock_session_instance.client.assert_called_once_with("bedrock-runtime")

    def test_convert_message_for_bedrock(self) -> None:
        """
        Test message conversion for Bedrock.
        """
        with patch("boto3.Session"):
            llm = BedrockAnthropicLLM(model_name="test-model")

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

            result = llm._convert_message_for_bedrock(message)

            self.assertEqual(result["role"], "user")
            self.assertEqual(len(result["content"]), 2)
            self.assertEqual(result["content"][0]["type"], "text")
            self.assertEqual(result["content"][1]["type"], "image")

    @patch("boto3.Session")
    def test_run_non_streaming_with_thinking(self, mock_session: Mock) -> None:
        """
        Test non-streaming run with thinking enabled.
        """
        mock_session_instance = Mock()
        mock_client = Mock()
        mock_session_instance.client.return_value = mock_client
        mock_session.return_value = mock_session_instance

        # Mock Bedrock response
        mock_response = {"body": Mock()}
        response_body = {
            "content": [
                {"type": "thinking", "thinking": "Let me think..."},
                {"type": "text", "text": "Final answer"},
            ]
        }
        mock_response["body"].read.return_value = json.dumps(response_body).encode()
        mock_client.invoke_model.return_value = mock_response

        params = ModelParameters(thinking_enabled=True, thinking_budget_tokens=1000)
        llm = BedrockAnthropicLLM(
            model_name="anthropic.claude-3-5-sonnet-20241022-v2:0", parameters=params
        )
        messages = [{"role": "user", "content": "Hello"}]

        result = llm._run_non_streaming(messages, [])

        self.assertEqual(result, "Final answer")

        # Verify the request body included thinking config
        call_args = mock_client.invoke_model.call_args
        request_body = json.loads(call_args[1]["body"])
        self.assertIn("thinking", request_body)
        self.assertEqual(request_body["thinking"]["type"], "enabled")
        self.assertEqual(request_body["thinking"]["budget_tokens"], 1000)

    @patch("boto3.Session")
    async def test_process_stream(self, mock_session: Mock) -> None:
        """
        Test stream processing.
        """
        mock_session_instance = Mock()
        mock_client = Mock()
        mock_session_instance.client.return_value = mock_client
        mock_session.return_value = mock_session_instance

        # Mock streaming response
        def mock_stream_events() -> iter:
            """
            Mock generator to simulate streaming events.
            """
            yield {
                "chunk": {
                    "bytes": (
                        json.dumps(
                            {"type": "content_block_delta", "delta": {"text": "Hello"}}
                        ).encode()
                    )
                }
            }
            yield {
                "chunk": {
                    "bytes": (
                        json.dumps(
                            {"type": "content_block_delta", "delta": {"text": " world"}}
                        ).encode()
                    )
                }
            }

        mock_response = {"body": mock_stream_events()}
        mock_client.invoke_model_with_response_stream.return_value = mock_response

        llm = BedrockAnthropicLLM(model_name="test-model")
        messages = [{"role": "user", "content": "Hello"}]

        stream = await llm._process_stream(messages, [])

        # Collect stream results
        results = []
        for chunk in stream:
            results.append(chunk)

        self.assertEqual(len(results), 2)


if __name__ == "__main__":
    unittest.main()
