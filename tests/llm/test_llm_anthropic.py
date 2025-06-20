"""
Test cases for AnthropicLLM class.
"""

import unittest
from unittest.mock import Mock, patch

from elemental_agents.llm.data_model import ModelParameters
from elemental_agents.llm.llm_anthropic import AnthropicLLM


class TestAnthropicLLM(unittest.TestCase):
    """
    Test Anthropic LLM implementation.
    """

    @patch("elemental_agents.llm.llm_anthropic.Anthropic")
    def test_anthropic_initialization(self, mock_anthropic: Mock) -> None:
        """
        Test Anthropic LLM initialization.
        """
        llm = AnthropicLLM(model_name="claude-3-5-sonnet-20241022", api_key="test_key")

        self.assertEqual(llm._model, "claude-3-5-sonnet-20241022")
        mock_anthropic.assert_called_once_with(api_key="test_key")

    def test_extract_system_message_empty(self) -> None:
        """
        Test extracting system message from empty list.
        """
        with patch("elemental_agents.llm.llm_anthropic.Anthropic"):
            llm = AnthropicLLM(model_name="claude-3-5-sonnet-20241022")

            system_msg, remaining = llm._extract_system_message([])

            self.assertEqual(system_msg, "")
            self.assertEqual(remaining, [])

    def test_extract_system_message_no_system(self) -> None:
        """
        Test extracting system message when none exists.
        """
        with patch("elemental_agents.llm.llm_anthropic.Anthropic"):
            llm = AnthropicLLM(model_name="claude-3-5-sonnet-20241022")

            messages = [{"role": "user", "content": "Hello"}]
            system_msg, remaining = llm._extract_system_message(messages)

            self.assertEqual(system_msg, "")
            self.assertEqual(remaining, messages)

    def test_convert_message_string_content(self) -> None:
        """
        Test converting message with string content.
        """
        with patch("elemental_agents.llm.llm_anthropic.Anthropic"):
            llm = AnthropicLLM(model_name="claude-3-5-sonnet-20241022")

            message = {"role": "user", "content": "Hello world"}
            result = llm._convert_message_for_anthropic(message)

            self.assertEqual(result, message)

    def test_convert_message_with_file_path_image(self) -> None:
        """
        Test converting message with file path image.
        """
        with patch("elemental_agents.llm.llm_anthropic.Anthropic"):
            llm = AnthropicLLM(model_name="claude-3-5-sonnet-20241022")

            message = {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello"},
                    {
                        "type": "image",
                        "file_path": "/path/to/image.png",
                        "media_type": "image/png",
                    },
                ],
            }

            with patch("os.path.exists", return_value=True):
                with patch("builtins.open", mock_open_image()):
                    result = llm._convert_message_for_anthropic(message)

            self.assertEqual(result["role"], "user")
            self.assertEqual(len(result["content"]), 2)
            self.assertEqual(result["content"][1]["type"], "image")
            self.assertIn("source", result["content"][1])

    @patch("elemental_agents.llm.llm_anthropic.Anthropic")
    def test_run_non_streaming_simple(self, mock_anthropic: Mock) -> None:
        """
        Test simple non-streaming run.
        """
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = "Simple response"
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client

        llm = AnthropicLLM(model_name="claude-3-5-sonnet-20241022")
        messages = [{"role": "user", "content": "Hello"}]

        result = llm._run_non_streaming(messages, [])

        self.assertEqual(result, "Simple response")

    @patch("elemental_agents.llm.llm_anthropic.Anthropic")
    def test_run_non_streaming_with_system_message(self, mock_anthropic: Mock) -> None:
        """
        Test non-streaming run with system message.
        """
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = "Response with system"
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client

        llm = AnthropicLLM(model_name="claude-3-5-sonnet-20241022")
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
        ]

        result = llm._run_non_streaming(messages, ["STOP"])

        self.assertEqual(result, "Response with system")

        # Verify system message was passed correctly
        call_args = mock_client.messages.create.call_args
        self.assertEqual(call_args[1]["system"], "You are helpful")
        self.assertEqual(call_args[1]["stop_sequences"], ["STOP"])

    def test_extract_content_from_chunk_invalid(self) -> None:
        """
        Test extracting content from invalid chunk.
        """
        with patch("elemental_agents.llm.llm_anthropic.Anthropic"):
            llm = AnthropicLLM(model_name="claude-3-5-sonnet-20241022")

            # Test with None
            result = llm._extract_content_from_chunk(None)
            self.assertIsNone(result)

            # Test with chunk without type
            chunk = Mock()
            del chunk.type  # Remove type attribute
            result = llm._extract_content_from_chunk(chunk)
            self.assertIsNone(result)

    def test_extract_content_from_chunk_unknown_type(self) -> None:
        """
        Test extracting content from chunk with unknown type.
        """
        with patch("elemental_agents.llm.llm_anthropic.Anthropic"):
            llm = AnthropicLLM(model_name="claude-3-5-sonnet-20241022")

            chunk = Mock()
            chunk.type = "unknown_type"

            result = llm._extract_content_from_chunk(chunk)
            self.assertIsNone(result)


def mock_open_image() -> Mock:
    """
    Mock open function for image files.
    """
    from unittest.mock import mock_open

    return mock_open(read_data=b"fake_image_data")


if __name__ == "__main__":
    unittest.main()
