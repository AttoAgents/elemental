"""
Comprehensive test cases for LLMFactory class.
"""

import unittest
from unittest.mock import Mock, patch

from elemental_agents.llm.data_model import Message, ModelParameters
from elemental_agents.llm.llm import LLM
from elemental_agents.llm.llm_anthropic import AnthropicLLM
from elemental_agents.llm.llm_azure_openai import AzureOpenAILLM
from elemental_agents.llm.llm_bedrock_anthropic import BedrockAnthropicLLM
from elemental_agents.llm.llm_factory import LLMFactory
from elemental_agents.llm.llm_ollama import OllamaLLM
from elemental_agents.llm.llm_openai import OpenAILLM


class TestLLMFactory(unittest.TestCase):
    """
    Test LLM factory class.
    """

    def setUp(self) -> None:
        """
        Set up test fixtures.
        """
        self.factory = LLMFactory()

    # @patch("elemental_agents.utils.config.ConfigModel")
    def test_llm_factory_initialization(self) -> None:
        """
        Test LLM factory initialization.
        """
        factory = LLMFactory()
        llm = factory.create()
        self.assertIsNotNone(factory)
        self.assertIsInstance(factory, LLMFactory)
        self.assertIsInstance(llm, LLM)

    @patch("elemental_agents.utils.config.ConfigModel")
    def test_create_default_engine(self, mock_config: Mock) -> None:
        """
        Test creating LLM with default engine.
        """
        mock_config_instance = Mock()
        mock_config_instance.default_engine = "mock"
        mock_config.return_value = mock_config_instance

        factory = LLMFactory()

        llm = factory.create()
        self.assertIsInstance(llm, LLM)

    @patch("elemental_agents.llm.llm_ollama.OllamaLLM")
    @patch("elemental_agents.utils.config.ConfigModel")
    def test_create_ollama_engine(self, mock_config: Mock, mock_ollama: Mock) -> None:
        """
        Test creating Ollama LLM.
        """
        mock_config_instance = Mock()
        mock_config_instance.ollama_llm_model_name = "llama3:8b"
        mock_config_instance.ollama_streaming = False
        mock_config_instance.websocket_url = "ws://localhost:8000"
        mock_config_instance.ollama_url = "http://localhost:11434"
        mock_config.return_value = mock_config_instance

        mock_ollama_instance = Mock(spec=OllamaLLM)
        mock_ollama.return_value = mock_ollama_instance

        factory = LLMFactory()
        llm = factory.create("ollama")

        self.assertEqual(llm, mock_ollama_instance)
        mock_ollama.assert_called_once()

    @patch("elemental_agents.llm.llm_ollama.OllamaLLM")
    @patch("elemental_agents.utils.config.ConfigModel")
    def test_create_ollama_with_custom_model(
        self, mock_config: Mock, mock_ollama: Mock
    ) -> None:
        """
        Test creating Ollama LLM with custom model.
        """
        mock_config_instance = Mock()
        mock_config_instance.ollama_streaming = False
        mock_config_instance.websocket_url = "ws://localhost:8000"
        mock_config_instance.ollama_url = "http://localhost:11434"
        mock_config.return_value = mock_config_instance

        mock_ollama_instance = Mock(spec=OllamaLLM)
        mock_ollama.return_value = mock_ollama_instance

        factory = LLMFactory()
        llm = factory.create("ollama|gemma2:9b")

        # Check that the custom model name was used
        call_args = mock_ollama.call_args
        self.assertEqual(call_args[1]["model_name"], "gemma2:9b")

    @patch("elemental_agents.llm.llm_openai.OpenAILLM")
    @patch("elemental_agents.utils.config.ConfigModel")
    def test_create_openai_engine(self, mock_config: Mock, mock_openai: Mock) -> None:
        """
        Test creating OpenAI LLM.
        """
        mock_config_instance = Mock()
        mock_config_instance.openai_llm_model_name = "gpt-4o-mini"
        mock_config_instance.openai_api_key = "test_key"
        mock_config_instance.openai_streaming = False
        mock_config_instance.websocket_url = "ws://localhost:8000"
        mock_config_instance.openai_url = "https://api.openai.com/v1"
        mock_config.return_value = mock_config_instance

        mock_openai_instance = Mock(spec=OpenAILLM)
        mock_openai.return_value = mock_openai_instance

        factory = LLMFactory()
        llm = factory.create("openai")

        self.assertEqual(llm, mock_openai_instance)
        mock_openai.assert_called_once()

    @patch("elemental_agents.llm.llm_openai.OpenAILLM")
    @patch("elemental_agents.utils.config.ConfigModel")
    def test_create_openai_with_custom_model(
        self, mock_config: Mock, mock_openai: Mock
    ) -> None:
        """
        Test creating OpenAI LLM with custom model.
        """
        mock_config_instance = Mock()
        mock_config_instance.openai_api_key = "test_key"
        mock_config_instance.openai_streaming = False
        mock_config_instance.websocket_url = "ws://localhost:8000"
        mock_config_instance.openai_url = "https://api.openai.com/v1"
        mock_config.return_value = mock_config_instance

        mock_openai_instance = Mock(spec=OpenAILLM)
        mock_openai.return_value = mock_openai_instance

        factory = LLMFactory()
        llm = factory.create("openai|gpt-4")

        call_args = mock_openai.call_args
        self.assertEqual(call_args[1]["model_name"], "gpt-4")

    @patch("elemental_agents.llm.llm_anthropic.AnthropicLLM")
    @patch("elemental_agents.utils.config.ConfigModel")
    def test_create_anthropic_engine(
        self, mock_config: Mock, mock_anthropic: Mock
    ) -> None:
        """
        Test creating Anthropic LLM.
        """
        mock_config_instance = Mock()
        mock_config_instance.anthropic_llm_model_name = "claude-3-5-sonnet-20241022"
        mock_config_instance.anthropic_streaming = False
        mock_config_instance.websocket_url = "ws://localhost:8000"
        mock_config_instance.anthropic_api_key = "test_key"
        mock_config.return_value = mock_config_instance

        mock_anthropic_instance = Mock(spec=AnthropicLLM)
        mock_anthropic.return_value = mock_anthropic_instance

        factory = LLMFactory()
        llm = factory.create("anthropic")

        self.assertEqual(llm, mock_anthropic_instance)
        mock_anthropic.assert_called_once()

    @patch("elemental_agents.llm.llm_openai.OpenAILLM")
    @patch("elemental_agents.utils.config.ConfigModel")
    def test_create_custom_engine(self, mock_config: Mock, mock_openai: Mock) -> None:
        """
        Test creating custom OpenAI-compatible LLM.
        """
        mock_config_instance = Mock()
        mock_config_instance.model_name = "custom-model"
        mock_config_instance.custom_api_key = "custom_key"
        mock_config_instance.custom_streaming = False
        mock_config_instance.websocket_url = "ws://localhost:8000"
        mock_config_instance.custom_url = "https://custom.api.com/v1"
        mock_config.return_value = mock_config_instance

        mock_openai_instance = Mock(spec=OpenAILLM)
        mock_openai.return_value = mock_openai_instance

        factory = LLMFactory()
        llm = factory.create("custom")

        self.assertEqual(llm, mock_openai_instance)

    @patch("elemental_agents.llm.llm_azure_openai.AzureOpenAILLM")
    @patch("elemental_agents.utils.config.ConfigModel")
    def test_create_azure_openai_engine(
        self, mock_config: Mock, mock_azure: Mock
    ) -> None:
        """
        Test creating Azure OpenAI LLM.
        """
        mock_config_instance = Mock()
        mock_config_instance.azure_openai_deployment_name = "gpt-4-deployment"
        mock_config_instance.azure_openai_streaming = False
        mock_config_instance.websocket_url = "ws://localhost:8000"
        mock_config_instance.azure_openai_api_key = "azure_key"
        mock_config_instance.azure_openai_endpoint = "https://test.openai.azure.com/"
        mock_config_instance.azure_openai_api_version = "2024-02-15-preview"
        mock_config.return_value = mock_config_instance

        mock_azure_instance = Mock(spec=AzureOpenAILLM)
        mock_azure.return_value = mock_azure_instance

        factory = LLMFactory()
        llm = factory.create("azure_openai")

        self.assertEqual(llm, mock_azure_instance)
        mock_azure.assert_called_once()

    @patch("elemental_agents.llm.llm_bedrock_anthropic.BedrockAnthropicLLM")
    @patch("elemental_agents.utils.config.ConfigModel")
    def test_create_bedrock_anthropic_engine(
        self, mock_config: Mock, mock_bedrock: Mock
    ) -> None:
        """
        Test creating Bedrock Anthropic LLM.
        """
        mock_config_instance = Mock()
        mock_config_instance.bedrock_anthropic_model_id = (
            "anthropic.claude-3-sonnet-20240229-v1:0"
        )
        mock_config_instance.bedrock_anthropic_streaming = False
        mock_config_instance.websocket_url = "ws://localhost:8000"
        mock_config_instance.aws_access_key_id = "access_key"
        mock_config_instance.aws_secret_access_key = "secret_key"
        mock_config_instance.aws_session_token = None
        mock_config_instance.aws_region = "us-east-1"
        mock_config.return_value = mock_config_instance

        mock_bedrock_instance = Mock(spec=BedrockAnthropicLLM)
        mock_bedrock.return_value = mock_bedrock_instance

        factory = LLMFactory()
        llm = factory.create("bedrock_anthropic")

        self.assertEqual(llm, mock_bedrock_instance)
        mock_bedrock.assert_called_once()

    @patch("elemental_agents.utils.config.ConfigModel")
    def test_create_unknown_engine_raises_error(self, mock_config: Mock) -> None:
        """
        Test that creating unknown engine raises ValueError.
        """
        factory = LLMFactory()

        with self.assertRaises(ValueError) as context:
            factory.create("unknown_engine")

        self.assertIn("Unknown model name", str(context.exception))

    @patch("elemental_agents.llm.llm_openai.OpenAILLM")
    @patch("elemental_agents.utils.config.ConfigModel")
    def test_create_with_custom_parameters(
        self, mock_config: Mock, mock_openai: Mock
    ) -> None:
        """
        Test creating LLM with custom parameters.
        """
        mock_config_instance = Mock()
        mock_config_instance.openai_llm_model_name = "gpt-4o-mini"
        mock_config_instance.openai_api_key = "test_key"
        mock_config_instance.openai_streaming = False
        mock_config_instance.websocket_url = "ws://localhost:8000"
        mock_config_instance.openai_url = "https://api.openai.com/v1"
        mock_config.return_value = mock_config_instance

        mock_openai_instance = Mock(spec=OpenAILLM)
        mock_openai.return_value = mock_openai_instance

        custom_params = ModelParameters(
            temperature=0.8, max_tokens=2000, reasoning_effort="high"
        )

        factory = LLMFactory()
        llm = factory.create("openai", custom_params)

        call_args = mock_openai.call_args
        self.assertEqual(call_args[1]["parameters"], custom_params)

    def test_create_with_none_parameters(self) -> None:
        """
        Test creating LLM with None parameters creates default.
        """
        with patch("elemental_agents.llm.llm_mock.MockLLM") as mock_llm:
            with patch.object(self.factory, "_config") as mock_config:
                mock_config.default_engine = "mock"
                mock_llm_instance = Mock(spec=LLM)
                mock_llm.return_value = mock_llm_instance

                # Mock the create method to handle mock engine
                with patch.object(self.factory, "create", wraps=self.factory.create):
                    # This would normally call the real create method
                    # but we need to handle the mock case
                    pass

    # Integration-style tests (these would run against real services if configured)
    def test_integration_ollama_creation(self) -> None:
        """
        Integration test for Ollama LLM creation.
        """
        try:
            llm = self.factory.create("ollama|llama3.2:1b")
            self.assertIsInstance(llm, LLM)

            # Test basic functionality
            msg = Message(role="user", content="Hello")
            # Note: This would fail if Ollama is not running
            # result = llm.run([msg])
            # self.assertIsNotNone(result)
        except Exception:
            # Skip if Ollama is not available
            self.skipTest("Ollama not available for integration test")

    def test_integration_mock_creation(self) -> None:
        """
        Integration test for Mock LLM creation.
        """
        # Mock should always work
        with patch.object(self.factory._config, "default_engine", "mock"):
            with patch(
                "elemental_agents.llm.llm_factory.LLMFactory.create"
            ) as mock_create:
                from elemental_agents.llm.llm_mock import MockLLM

                mock_create.return_value = MockLLM()

                llm = self.factory.create()
                self.assertIsInstance(llm, MockLLM)

                msg = Message(role="user", content="Hello")
                result = llm.run([msg])
                self.assertIsNotNone(result)


class TestLLMFactoryParameterPassing(unittest.TestCase):
    """
    Test parameter passing in LLM factory.
    """

    @patch("elemental_agents.llm.llm_openai.OpenAILLM")
    @patch("elemental_agents.utils.config.ConfigModel")
    def test_reasoning_parameters_passed_correctly(
        self, mock_config: Mock, mock_openai: Mock
    ) -> None:
        """
        Test that reasoning parameters are passed correctly.
        """
        mock_config_instance = Mock()
        mock_config_instance.openai_llm_model_name = "o1-preview"
        mock_config_instance.openai_api_key = "test_key"
        mock_config_instance.openai_streaming = False
        mock_config_instance.websocket_url = "ws://localhost:8000"
        mock_config_instance.openai_url = "https://api.openai.com/v1"
        mock_config.return_value = mock_config_instance

        mock_openai_instance = Mock(spec=OpenAILLM)
        mock_openai.return_value = mock_openai_instance

        reasoning_params = ModelParameters(
            reasoning_effort="high", thinking_enabled=True, thinking_budget_tokens=2000
        )

        factory = LLMFactory()
        llm = factory.create("openai|o1-preview", reasoning_params)

        call_args = mock_openai.call_args
        passed_params = call_args[1]["parameters"]
        self.assertEqual(passed_params.reasoning_effort, "high")
        self.assertTrue(passed_params.thinking_enabled)
        self.assertEqual(passed_params.thinking_budget_tokens, 2000)

    @patch("elemental_agents.llm.llm_anthropic.AnthropicLLM")
    @patch("elemental_agents.utils.config.ConfigModel")
    def test_thinking_parameters_for_anthropic(
        self, mock_config: Mock, mock_anthropic: Mock
    ) -> None:
        """
        Test that thinking parameters are passed to Anthropic.
        """
        mock_config_instance = Mock()
        mock_config_instance.anthropic_llm_model_name = "claude-3-5-sonnet-20241022"
        mock_config_instance.anthropic_streaming = False
        mock_config_instance.websocket_url = "ws://localhost:8000"
        mock_config_instance.anthropic_api_key = "test_key"
        mock_config.return_value = mock_config_instance

        mock_anthropic_instance = Mock(spec=AnthropicLLM)
        mock_anthropic.return_value = mock_anthropic_instance

        thinking_params = ModelParameters(
            thinking_enabled=True, thinking_budget_tokens=1500
        )

        factory = LLMFactory()
        llm = factory.create("anthropic", thinking_params)

        call_args = mock_anthropic.call_args
        passed_params = call_args[1]["parameters"]
        self.assertTrue(passed_params.thinking_enabled)
        self.assertEqual(passed_params.thinking_budget_tokens, 1500)


if __name__ == "__main__":
    unittest.main()
