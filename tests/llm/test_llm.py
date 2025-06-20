"""
Comprehensive test cases for LLM classes.
"""

import asyncio
import unittest
from unittest.mock import Mock, patch

from elemental_agents.llm.data_model import Message, ModelParameters
from elemental_agents.llm.llm_anthropic import AnthropicLLM
from elemental_agents.llm.llm_mock import MockLLM
from elemental_agents.llm.llm_ollama import OllamaLLM
from elemental_agents.llm.llm_openai import OpenAILLM

MODEL_NAME = "llama3:8b"
OPENAI_MODEL = "gpt-4o-mini"
ANTHROPIC_MODEL = "claude-3-5-sonnet-20241022"


class TestLLMBase(unittest.TestCase):
    """
    Test base LLM functionality.
    """

    def test_llm_initialization(self) -> None:
        """
        Test LLM base class initialization.
        """
        params = ModelParameters(temperature=0.5, max_tokens=500)
        llm = MockLLM(parameters=params)

        self.assertEqual(llm._temperature, 0.5)
        self.assertEqual(llm._max_tokens, 500)
        self.assertEqual(llm._stop, [])
        self.assertEqual(llm._max_retries, 3)

    def test_llm_set_temperature(self) -> None:
        """
        Test setting temperature.
        """
        llm = MockLLM()
        llm.set_temperature(0.8)

        self.assertEqual(llm._temperature, 0.8)

    def test_llm_stop_words_initialization(self) -> None:
        """
        Test stop words initialization.
        """
        params = ModelParameters(stop=["STOP", "END"])
        llm = MockLLM(parameters=params)

        self.assertEqual(llm._stop, ["STOP", "END"])

    def test_llm_reasoning_parameters(self) -> None:
        """
        Test reasoning parameters initialization.
        """
        params = ModelParameters(
            reasoning_effort="high", thinking_enabled=True, thinking_budget_tokens=2000
        )
        llm = MockLLM(parameters=params)

        self.assertEqual(llm._reasoning_effort, "high")
        self.assertTrue(llm._thinking_enabled)
        self.assertEqual(llm._thinking_budget_tokens, 2000)


class TestMockLLM(unittest.TestCase):
    """
    Test MockLLM class for testing purposes.
    """

    def test_run_with_empty_messages(self) -> None:
        """
        Test MockLLM with empty messages.
        """
        llm = MockLLM()
        result = llm.run([])

        self.assertEqual(result, "No messages to process.")

    def test_run_with_user_message(self) -> None:
        """
        Test MockLLM with a user message.
        """
        llm = MockLLM()
        messages = [Message(role="user", content="Hello")]
        result = llm.run(messages)

        self.assertEqual(result, "Output for user message (Hello)")

    def test_run_with_system_message(self) -> None:
        """
        Test MockLLM with a system message.
        """
        llm = MockLLM()
        messages = [Message(role="system", content="Welcome")]
        result = llm.run(messages)

        self.assertEqual(result, "Output for system message (Welcome)")

    def test_run_with_multiple_messages(self) -> None:
        """
        Test MockLLM with multiple messages.
        """
        llm = MockLLM()
        messages = [
            Message(role="user", content="Hello"),
            Message(role="system", content="How can I help you?"),
        ]
        result = llm.run(messages)

        self.assertEqual(result, "Output for system message (How can I help you?)")

    def test_run_with_stop_word(self) -> None:
        """
        Test MockLLM with stop word.
        """
        llm = MockLLM()
        messages = [Message(role="user", content="Hello")]
        result = llm.run(messages, stop_word="STOP")

        self.assertIsNotNone(result)

    def test_run_with_structured_content(self) -> None:
        """
        Test MockLLM with structured content.
        """
        llm = MockLLM()
        message = Message(role="user", content="Hello")
        message.add_image(b"fake_image_data")
        result = llm.run([message])

        self.assertIsNotNone(result)

    def test_extract_content_from_chunk(self) -> None:
        """
        Test extracting content from chunk.
        """
        llm = MockLLM()
        chunk = {"content": "test content"}

        result = llm._extract_content_from_chunk(chunk)
        self.assertEqual(result, "test content")

    def test_extract_content_from_empty_chunk(self) -> None:
        """
        Test extracting content from empty chunk.
        """
        llm = MockLLM()

        result = llm._extract_content_from_chunk({})
        self.assertIsNone(result)

    def test_extract_content_from_none_chunk(self) -> None:
        """
        Test extracting content from None chunk.
        """
        llm = MockLLM()

        result = llm._extract_content_from_chunk(None)
        self.assertIsNone(result)

    def test_process_stream(self) -> None:
        """
        Test stream processing.
        """

        async def run_test() -> None:
            """
            Run the async test for stream processing.
            """
            llm = MockLLM()
            messages = [{"role": "user", "content": "Hello"}]

            stream = await llm._process_stream(messages, [])

            # Collect all chunks
            chunks = []
            async for chunk in stream:
                chunks.append(chunk)

            self.assertGreater(len(chunks), 0)

        # Run the async test
        asyncio.run(run_test())


class TestOpenAILLM(unittest.TestCase):
    """
    Test OpenAI LLM implementation.
    """

    @patch("elemental_agents.llm.llm_openai.OpenAI")
    def test_openai_llm_initialization(self, mock_openai: Mock) -> None:
        """
        Test OpenAI LLM initialization.
        """
        llm = OpenAILLM(
            model_name=OPENAI_MODEL,
            openai_api_key="test_key",
            url="https://api.openai.com/v1",
        )

        self.assertEqual(llm._model, OPENAI_MODEL)
        mock_openai.assert_called_once_with(
            api_key="test_key", base_url="https://api.openai.com/v1"
        )

    @patch("elemental_agents.llm.llm_openai.OpenAI")
    def test_convert_message_for_openai_string_content(self, mock_openai: Mock) -> None:
        """
        Test converting string message for OpenAI.
        """
        # Mock the OpenAI client
        mock_openai.return_value = Mock()

        llm = OpenAILLM(model_name=OPENAI_MODEL, openai_api_key="test_key")
        message = {"role": "user", "content": "Hello"}

        result = llm._convert_message_for_openai(message)

        self.assertEqual(result, message)

    @patch("elemental_agents.llm.llm_openai.OpenAI")
    def test_convert_message_for_openai_structured_content(self, mock_openai: Mock) -> None:
        """
        Test converting structured message for OpenAI.
        """
        # Mock the OpenAI client
        mock_openai.return_value = Mock()

        llm = OpenAILLM(model_name=OPENAI_MODEL, openai_api_key="test_key")
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

        result = llm._convert_message_for_openai(message)

        self.assertEqual(result["role"], "user")
        self.assertEqual(len(result["content"]), 2)
        self.assertEqual(result["content"][0]["type"], "text")
        self.assertEqual(result["content"][1]["type"], "image_url")

    @patch("elemental_agents.llm.llm_openai.OpenAI")
    def test_run_non_streaming(self, mock_openai: Mock) -> None:
        """
        Test non-streaming run.
        """
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Test response"
        mock_response.usage = Mock()
        mock_response.usage.total_tokens = 100
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        llm = OpenAILLM(model_name=OPENAI_MODEL, openai_api_key="test_key")
        messages = [{"role": "user", "content": "Hello"}]

        result = llm._run_non_streaming(messages, [])

        self.assertEqual(result, "Test response")
        mock_client.chat.completions.create.assert_called_once()

    @patch("elemental_agents.llm.llm_openai.OpenAI")
    def test_run_with_reasoning_effort(self, mock_openai: Mock) -> None:
        """
        Test run with reasoning effort for o1 models.
        """
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Test response"
        mock_response.usage = None
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        params = ModelParameters(reasoning_effort="high")
        llm = OpenAILLM(
            model_name="o1-preview", parameters=params, openai_api_key="test_key"
        )
        messages = [{"role": "user", "content": "Hello"}]

        result = llm._run_non_streaming(messages, [])

        self.assertEqual(result, "Test response")
        # Check that reasoning_effort was passed
        call_args = mock_client.chat.completions.create.call_args
        self.assertEqual(call_args[1]["reasoning_effort"], "high")

    @patch("elemental_agents.llm.llm_openai.OpenAI")
    def test_extract_content_from_chunk(self, mock_openai: Mock) -> None:
        """
        Test extracting content from OpenAI chunk.
        """
        # Mock the OpenAI client
        mock_openai.return_value = Mock()

        llm = OpenAILLM(model_name=OPENAI_MODEL, openai_api_key="test_key")

        # Mock chunk with content
        chunk = Mock()
        chunk.choices = [Mock()]
        chunk.choices[0].delta = Mock()
        chunk.choices[0].delta.content = "test content"

        result = llm._extract_content_from_chunk(chunk)
        self.assertEqual(result, "test content")

    @patch("elemental_agents.llm.llm_openai.OpenAI")
    def test_extract_content_from_empty_chunk(self, mock_openai: Mock) -> None:
        """
        Test extracting content from empty OpenAI chunk.
        """
        # Mock the OpenAI client
        mock_openai.return_value = Mock()

        llm = OpenAILLM(model_name=OPENAI_MODEL, openai_api_key="test_key")

        chunk = Mock()
        chunk.choices = []

        result = llm._extract_content_from_chunk(chunk)
        self.assertIsNone(result)


class TestAnthropicLLM(unittest.TestCase):
    """
    Test Anthropic LLM implementation.
    """

    @patch("elemental_agents.llm.llm_anthropic.Anthropic")
    def test_anthropic_llm_initialization(self, mock_anthropic: Mock) -> None:
        """
        Test Anthropic LLM initialization.
        """
        llm = AnthropicLLM(model_name=ANTHROPIC_MODEL, api_key="test_key")

        self.assertEqual(llm._model, ANTHROPIC_MODEL)
        mock_anthropic.assert_called_once_with(api_key="test_key")

    @patch("elemental_agents.llm.llm_anthropic.Anthropic")
    def test_extract_system_message(self, mock_anthropic: Mock) -> None:
        """
        Test extracting system message.
        """
        llm = AnthropicLLM(model_name=ANTHROPIC_MODEL, api_key="test_key")
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
        ]

        system_msg, remaining = llm._extract_system_message(messages)

        self.assertEqual(system_msg, "You are helpful")
        self.assertEqual(len(remaining), 1)
        self.assertEqual(remaining[0]["role"], "user")

    @patch("elemental_agents.llm.llm_anthropic.Anthropic")
    def test_extract_system_message_structured(self, mock_anthropic: Mock) -> None:
        """
        Test extracting structured system message.
        """
        llm = AnthropicLLM(model_name=ANTHROPIC_MODEL, api_key="test_key")
        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are"},
                    {"type": "text", "text": "helpful"},
                ],
            },
            {"role": "user", "content": "Hello"},
        ]

        system_msg, remaining = llm._extract_system_message(messages)

        self.assertEqual(system_msg, "You are helpful")
        self.assertEqual(len(remaining), 1)

    @patch("elemental_agents.llm.llm_anthropic.Anthropic")
    def test_convert_message_for_anthropic(self, mock_anthropic: Mock) -> None:
        """
        Test converting message for Anthropic.
        """
        llm = AnthropicLLM(model_name=ANTHROPIC_MODEL, api_key="test_key")
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

        result = llm._convert_message_for_anthropic(message)

        self.assertEqual(result["role"], "user")
        self.assertEqual(len(result["content"]), 2)
        self.assertEqual(result["content"][0]["type"], "text")
        self.assertEqual(result["content"][1]["type"], "image")

    @patch("elemental_agents.llm.llm_anthropic.Anthropic")
    def test_run_with_thinking_enabled(self, mock_anthropic: Mock) -> None:
        """
        Test run with thinking enabled.
        """
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = [
            Mock(type="thinking", thinking="Let me think..."),
            Mock(type="text", text="Final answer"),
        ]
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client

        params = ModelParameters(thinking_enabled=True, thinking_budget_tokens=1000)
        llm = AnthropicLLM(
            model_name="claude-3-5-sonnet-20241022",
            parameters=params,
            api_key="test_key",
        )
        messages = [{"role": "user", "content": "Hello"}]

        result = llm._run_non_streaming(messages, [])

        self.assertEqual(result, "Final answer")
        # Check that thinking was passed
        call_args = mock_client.messages.create.call_args
        self.assertIn("thinking", call_args[1])

    @patch("elemental_agents.llm.llm_anthropic.Anthropic")
    def test_extract_content_from_chunk(self, mock_anthropic: Mock) -> None:
        """
        Test extracting content from Anthropic chunk.
        """
        llm = AnthropicLLM(model_name=ANTHROPIC_MODEL, api_key="test_key")

        # Test content_block_delta
        chunk = Mock()
        chunk.type = "content_block_delta"
        chunk.delta = Mock()
        chunk.delta.text = "test content"

        result = llm._extract_content_from_chunk(chunk)
        self.assertEqual(result, "test content")

    @patch("elemental_agents.llm.llm_anthropic.Anthropic")
    def test_extract_content_from_chunk_start(self, mock_anthropic: Mock) -> None:
        """
        Test extracting content from Anthropic chunk start.
        """
        llm = AnthropicLLM(model_name=ANTHROPIC_MODEL, api_key="test_key")

        chunk = Mock()
        chunk.type = "content_block_start"
        chunk.content_block = Mock()
        chunk.content_block.text = "start content"

        result = llm._extract_content_from_chunk(chunk)
        self.assertEqual(result, "start content")


class TestOllamaLLM(unittest.TestCase):
    """
    Test Ollama LLM implementation.
    """

    @patch("elemental_agents.llm.llm_ollama.Client")
    def test_ollama_llm_initialization(self, mock_client: Mock) -> None:
        """
        Test Ollama LLM initialization.
        """
        llm = OllamaLLM(model_name=MODEL_NAME, url="http://localhost:11434")

        self.assertEqual(llm._model, MODEL_NAME)
        mock_client.assert_called_once_with(host="http://localhost:11434")

    @patch("elemental_agents.llm.llm_ollama.Client")
    def test_prepare_options(self, mock_client: Mock) -> None:
        """
        Test preparing options for Ollama.
        """
        params = ModelParameters(
            temperature=0.7,
            max_tokens=500,
            frequency_penalty=0.1,
            presence_penalty=0.2,
            top_p=0.9,
        )
        llm = OllamaLLM(model_name=MODEL_NAME, parameters=params)

        options = llm._prepare_options(["STOP"])

        self.assertEqual(options["temperature"], 0.7)
        self.assertEqual(options["num_predict"], 500)
        self.assertEqual(options["stop"], ["STOP"])
        self.assertEqual(options["frequency_penalty"], 0.1)
        self.assertEqual(options["presence_penalty"], 0.2)
        self.assertEqual(options["top_p"], 0.9)

    @patch("elemental_agents.llm.llm_ollama.Client")
    def test_convert_message_for_ollama_string(self, mock_client: Mock) -> None:
        """
        Test converting string message for Ollama.
        """
        llm = OllamaLLM(model_name=MODEL_NAME)
        message = {"role": "user", "content": "Hello"}

        result = llm._convert_message_for_ollama(message)

        self.assertEqual(result, {"role": "user", "content": "Hello"})

    @patch("elemental_agents.llm.llm_ollama.Client")
    def test_convert_message_for_ollama_with_images(self, mock_client: Mock) -> None:
        """
        Test converting message with images for Ollama.
        """
        llm = OllamaLLM(model_name=MODEL_NAME)
        message = {
            "role": "user",
            "content": [
                {"type": "text", "text": "Hello"},
                {"type": "image", "file_path": "/path/to/image.png"},
            ],
        }

        with patch("os.path.exists", return_value=True):
            result = llm._convert_message_for_ollama(message)

        self.assertEqual(result["role"], "user")
        self.assertEqual(result["content"], "Hello")
        self.assertEqual(result["images"], ["/path/to/image.png"])

    @patch("elemental_agents.llm.llm_ollama.Client")
    def test_run_with_thinking_enabled(self, mock_client_class: Mock) -> None:
        """
        Test run with thinking enabled for supported models.
        """
        mock_client = Mock()
        mock_response = {
            "message": {"thinking": "Let me think...", "content": "Final answer"}
        }
        mock_client.chat.return_value = mock_response
        mock_client_class.return_value = mock_client

        params = ModelParameters(thinking_enabled=True)
        llm = OllamaLLM(model_name="deepseek-r1:latest", parameters=params)
        messages = [{"role": "user", "content": "Hello"}]

        result = llm._run_non_streaming(messages, [])

        self.assertEqual(result, "Final answer")
        # Check that think=True was passed
        call_args = mock_client.chat.call_args
        self.assertTrue(call_args[1]["think"])

    @patch("elemental_agents.llm.llm_ollama.Client")
    def test_extract_content_from_chunk(self, mock_client: Mock) -> None:
        """
        Test extracting content from Ollama chunk.
        """
        llm = OllamaLLM(model_name=MODEL_NAME)

        chunk = {"message": {"content": "test content"}}

        result = llm._extract_content_from_chunk(chunk)
        self.assertEqual(result, "test content")

    @patch("elemental_agents.llm.llm_ollama.Client")
    def test_extract_content_from_invalid_chunk(self, mock_client: Mock) -> None:
        """
        Test extracting content from invalid Ollama chunk.
        """
        llm = OllamaLLM(model_name=MODEL_NAME)

        result = llm._extract_content_from_chunk("invalid")
        self.assertIsNone(result)


class TestLLMRetryMechanism(unittest.TestCase):
    """
    Test LLM retry mechanism.
    """

    @patch("elemental_agents.llm.llm_mock.MockLLM._run_non_streaming")
    def test_retry_on_exception(self, mock_run: Mock) -> None:
        """
        Test that LLM retries on exceptions.
        """
        # First two calls fail, third succeeds
        mock_run.side_effect = [
            ConnectionError("Network error"),
            TimeoutError("Timeout"),
            "Success",
        ]

        llm = MockLLM(max_retries=3)
        messages = [Message(role="user", content="Hello")]

        result = llm.run(messages)

        self.assertEqual(result, "Success")
        self.assertEqual(mock_run.call_count, 3)

    @patch("elemental_agents.llm.llm_mock.MockLLM._run_non_streaming")
    def test_retry_exhausted(self, mock_run: Mock) -> None:
        """
        Test that LLM raises exception when retries are exhausted.
        """
        mock_run.side_effect = ConnectionError("Network error")

        llm = MockLLM(max_retries=2)
        messages = [Message(role="user", content="Hello")]

        with self.assertRaises(ConnectionError):
            llm.run(messages)

        self.assertEqual(mock_run.call_count, 2)


if __name__ == "__main__":
    unittest.main()
