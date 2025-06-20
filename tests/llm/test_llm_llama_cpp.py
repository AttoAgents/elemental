"""
Comprehensive test cases for LlamaCppLLM class.
"""

import unittest
from unittest.mock import Mock, patch

from elemental_agents.llm.data_model import Message, ModelParameters
from elemental_agents.llm.llm_llama_cpp import LlamaCppLLM


class TestLlamaCppLLM(unittest.TestCase):
    """
    Test the LlamaCpp LLM implementation.
    """

    @patch("elemental_agents.llm.llm_llama_cpp.Llama")
    def test_llama_cpp_initialization(self, mock_llama: Mock) -> None:
        """
        Test LlamaCppLLM initialization.
        """
        mock_llama_instance = Mock()
        mock_llama.return_value = mock_llama_instance

        parameters = ModelParameters(temperature=0.5, max_tokens=1000)
        llm = LlamaCppLLM(
            model_name="test_model.gguf", parameters=parameters, n_ctx=4096, n_threads=8
        )

        self.assertEqual(llm._model, "test_model.gguf")
        self.assertEqual(llm._temperature, 0.5)
        self.assertEqual(llm._max_tokens, 1000)

        mock_llama.assert_called_once_with(
            model_path="test_model.gguf", chat_format="chatml", n_ctx=4096, n_threads=8
        )

    @patch("elemental_agents.llm.llm_llama_cpp.Llama")
    def test_llama_cpp_initialization_defaults(self, mock_llama: Mock) -> None:
        """
        Test LlamaCppLLM initialization with defaults.
        """
        mock_llama_instance = Mock()
        mock_llama.return_value = mock_llama_instance

        llm = LlamaCppLLM(model_name="test_model.gguf")

        mock_llama.assert_called_once_with(
            model_path="test_model.gguf",
            chat_format="chatml",
            n_ctx=2048,
            n_threads=None,
        )

    def test_convert_message_for_llama(self) -> None:
        """
        Test message conversion for Llama CPP.
        """
        with patch("elemental_agents.llm.llm_llama_cpp.Llama"):
            llm = LlamaCppLLM(model_name="test_model.gguf")

            message = {"role": "user", "content": "Hello"}
            result = llm._convert_message_for_llama(message)

            # Llama CPP uses messages as-is
            self.assertEqual(result, message)

    @patch("elemental_agents.llm.llm_llama_cpp.Llama")
    def test_run_non_streaming(self, mock_llama: Mock) -> None:
        """
        Test non-streaming run.
        """
        mock_llama_instance = Mock()
        mock_response = {"choices": [{"message": {"content": "Test response"}}]}
        mock_llama_instance.create_chat_completion.return_value = mock_response
        mock_llama.return_value = mock_llama_instance

        parameters = ModelParameters(
            temperature=0.7,
            max_tokens=500,
            frequency_penalty=0.1,
            presence_penalty=0.2,
            top_p=0.9,
        )
        llm = LlamaCppLLM(model_name="test_model.gguf", parameters=parameters)

        messages = [{"role": "user", "content": "Hello"}]
        result = llm._run_non_streaming(messages, ["STOP"])

        self.assertEqual(result, "Test response")

        # Verify the call was made with correct parameters
        mock_llama_instance.create_chat_completion.assert_called_once_with(
            messages=messages,
            temperature=0.7,
            stop=["STOP"],
            max_tokens=500,
            presence_penalty=0.2,
            frequency_penalty=0.1,
            top_p=0.9,
            stream=False,
        )

    @patch("elemental_agents.llm.llm_llama_cpp.Llama")
    def test_run_non_streaming_error_handling(self, mock_llama: Mock) -> None:
        """
        Test error handling in non-streaming run.
        """
        mock_llama_instance = Mock()
        mock_llama_instance.create_chat_completion.side_effect = ValueError(
            "Model error"
        )
        mock_llama.return_value = mock_llama_instance

        llm = LlamaCppLLM(model_name="test_model.gguf")
        messages = [{"role": "user", "content": "Hello"}]

        with self.assertRaises(ValueError):
            llm._run_non_streaming(messages, [])

    @patch("elemental_agents.llm.llm_llama_cpp.Llama")
    async def test_process_stream(self, mock_llama: Mock) -> None:
        """
        Test stream processing.
        """
        mock_llama_instance = Mock()

        # Mock streaming response
        def mock_stream() -> Mock:
            """
            Mock generator to simulate streaming response.
            """
            yield {"choices": [{"delta": {"content": "Hello"}}]}
            yield {"choices": [{"delta": {"content": " world"}}]}
            yield {"choices": [{"delta": {"content": "!"}}]}

        mock_llama_instance.create_chat_completion.return_value = mock_stream()
        mock_llama.return_value = mock_llama_instance

        llm = LlamaCppLLM(model_name="test_model.gguf")
        messages = [{"role": "user", "content": "Hello"}]

        stream = await llm._process_stream(messages, [])

        # Collect stream results
        results = []
        for chunk in stream:
            results.append(chunk)

        self.assertEqual(len(results), 3)

        # Verify the call was made with streaming=True
        mock_llama_instance.create_chat_completion.assert_called_once_with(
            messages=messages,
            temperature=0.0,  # default
            stop=[],
            max_tokens=1000,  # default
            presence_penalty=0.0,  # default
            frequency_penalty=0.0,  # default
            top_p=1.0,  # default
            stream=True,
        )

    @patch("elemental_agents.llm.llm_llama_cpp.Llama")
    async def test_process_stream_error_handling(self, mock_llama: Mock) -> None:
        """
        Test error handling in stream processing.
        """
        mock_llama_instance = Mock()
        mock_llama_instance.create_chat_completion.side_effect = RuntimeError(
            "Stream error"
        )
        mock_llama.return_value = mock_llama_instance

        llm = LlamaCppLLM(model_name="test_model.gguf")
        messages = [{"role": "user", "content": "Hello"}]

        with self.assertRaises(RuntimeError):
            await llm._process_stream(messages, [])

    def test_extract_content_from_chunk(self) -> None:
        """
        Test extracting content from chunk.
        """
        with patch("elemental_agents.llm.llm_llama_cpp.Llama"):
            llm = LlamaCppLLM(model_name="test_model.gguf")

            # Valid chunk
            chunk = {"choices": [{"delta": {"content": "test content"}}]}

            result = llm._extract_content_from_chunk(chunk)
            self.assertEqual(result, "test content")

    def test_extract_content_from_chunk_no_content(self) -> None:
        """
        Test extracting content from chunk with no content.
        """
        with patch("elemental_agents.llm.llm_llama_cpp.Llama"):
            llm = LlamaCppLLM(model_name="test_model.gguf")

            # Chunk without content
            chunk = {"choices": [{"delta": {}}]}

            result = llm._extract_content_from_chunk(chunk)
            self.assertIsNone(result)

    def test_extract_content_from_empty_chunk(self) -> None:
        """
        Test extracting content from empty chunk.
        """
        with patch("elemental_agents.llm.llm_llama_cpp.Llama"):
            llm = LlamaCppLLM(model_name="test_model.gguf")

            # Empty chunk
            chunk = {"choices": []}

            result = llm._extract_content_from_chunk(chunk)
            self.assertIsNone(result)

    def test_extract_content_from_invalid_chunk(self) -> None:
        """Test extracting content from invalid chunk."""
        with patch("elemental_agents.llm.llm_llama_cpp.Llama"):
            llm = LlamaCppLLM(model_name="test_model.gguf")

            # Invalid chunk
            chunk = "invalid"

            result = llm._extract_content_from_chunk(chunk)
            self.assertIsNone(result)

    def test_extract_content_from_none_chunk(self) -> None:
        """Test extracting content from None chunk."""
        with patch("elemental_agents.llm.llm_llama_cpp.Llama"):
            llm = LlamaCppLLM(model_name="test_model.gguf")

            result = llm._extract_content_from_chunk(None)
            self.assertIsNone(result)

    @patch("elemental_agents.llm.llm_llama_cpp.Llama")
    def test_full_integration_mock(self, mock_llama: Mock) -> None:
        """
        Test full integration with mocked Llama.
        """
        mock_llama_instance = Mock()
        mock_response = {
            "choices": [
                {"message": {"content": "The sky is blue due to Rayleigh scattering."}}
            ]
        }
        mock_llama_instance.create_chat_completion.return_value = mock_response
        mock_llama.return_value = mock_llama_instance

        parameters = ModelParameters(temperature=0.0)
        llm = LlamaCppLLM(
            model_name="test_model.gguf",
            message_stream=False,
            parameters=parameters,
        )

        messages = [Message(content="Why is the sky blue?", role="user")]
        response = llm.run(messages=messages)

        self.assertIsInstance(response, str)
        self.assertEqual(response, "The sky is blue due to Rayleigh scattering.")
        self.assertGreater(len(response), 0)

    # This test would require an actual model file to run
    def test_real_model_integration(self) -> None:
        """
        Integration test with real model (skipped if model not available).
        """
        model_path = "models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"

        try:
            parameters = ModelParameters(temperature=0.0)
            llm = LlamaCppLLM(
                model_name=model_path,
                message_stream=False,
                parameters=parameters,
            )

            messages = [Message(content="Why is the sky blue?", role="user")]
            response = llm.run(messages=messages)

            self.assertIsInstance(response, str)
            self.assertGreater(len(response), 0)

        except Exception:
            self.skipTest(f"Model file {model_path} not available for integration test")


class TestLlamaCppLLMParameterHandling(unittest.TestCase):
    """Test parameter handling in LlamaCppLLM."""

    @patch("elemental_agents.llm.llm_llama_cpp.Llama")
    def test_all_parameters_passed_correctly(self, mock_llama: Mock) -> None:
        """
        Test that all parameters are passed correctly to Llama CPP.
        """
        mock_llama_instance = Mock()
        mock_response = {"choices": [{"message": {"content": "response"}}]}
        mock_llama_instance.create_chat_completion.return_value = mock_response
        mock_llama.return_value = mock_llama_instance

        parameters = ModelParameters(
            temperature=0.8,
            stop=["STOP", "END"],
            max_tokens=2000,
            frequency_penalty=0.5,
            presence_penalty=0.3,
            top_p=0.95,
        )

        llm = LlamaCppLLM(model_name="test_model.gguf", parameters=parameters)
        messages = [{"role": "user", "content": "Test"}]

        llm._run_non_streaming(messages, ["ADDITIONAL_STOP"])

        call_args = mock_llama_instance.create_chat_completion.call_args
        self.assertEqual(call_args[1]["temperature"], 0.8)
        self.assertEqual(call_args[1]["stop"], ["ADDITIONAL_STOP"])
        self.assertEqual(call_args[1]["max_tokens"], 2000)
        self.assertEqual(call_args[1]["frequency_penalty"], 0.5)
        self.assertEqual(call_args[1]["presence_penalty"], 0.3)
        self.assertEqual(call_args[1]["top_p"], 0.95)

    @patch("elemental_agents.llm.llm_llama_cpp.Llama")
    def test_context_and_thread_parameters(self, mock_llama: Mock) -> None:
        """
        Test context and thread parameters.
        """
        llm = LlamaCppLLM(model_name="test_model.gguf", n_ctx=8192, n_threads=16)

        mock_llama.assert_called_once_with(
            model_path="test_model.gguf", chat_format="chatml", n_ctx=8192, n_threads=16
        )


if __name__ == "__main__":
    unittest.main()
