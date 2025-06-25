"""
Language model factory class to create LLM instances.
"""

from typing import Optional

from loguru import logger

from elemental_agents.llm.data_model import ModelParameters
from elemental_agents.llm.llm import LLM
from elemental_agents.utils.config import ConfigModel


class LLMFactory:
    """
    Factory class for creating LLM instances with parameters from the configuration file.
    """

    def __init__(self) -> None:
        """
        Initialize the LLM factory with the configuration model.
        """
        self._config = ConfigModel()

    def create(
        self,
        engine_name: str = None,
        model_parameters: Optional[ModelParameters] = None,
    ) -> LLM:
        """
        Create an LLM instance based on the engine name. If the engine name is
        not provided, the default engine is used that is specified in the
        configuration file.

        :param engine_name: The name of the engine to use.
        :param model_parameters: The parameters for the LLM instance.
        :return: An instance of the LLM class.
        """
        if model_parameters is None:
            model_parameters = ModelParameters()

        llm_parameters = []

        if engine_name is None:
            local_engine_name = self._config.default_engine
        else:
            llm_parameters = engine_name.split("|")
            local_engine_name = llm_parameters[0]

        if local_engine_name == "ollama":

            from elemental_agents.llm.llm_ollama import OllamaLLM

            local_model_name = (
                llm_parameters[1]
                if len(llm_parameters) > 1
                else self._config.ollama_llm_model_name
            )

            logger.debug("Creating Ollama LLM instance.")
            logger.debug(f"Model name: {local_model_name}")

            ollama_llm = OllamaLLM(
                model_name=local_model_name,
                message_stream=self._config.ollama_streaming,
                stream_url=self._config.websocket_url,
                parameters=model_parameters,
                url=self._config.ollama_url,
            )
            return ollama_llm

        if local_engine_name == "openai":

            from elemental_agents.llm.llm_openai import OpenAILLM

            local_model_name = (
                llm_parameters[1]
                if len(llm_parameters) > 1
                else self._config.openai_llm_model_name
            )

            logger.debug("Creating OpenAI LLM instance.")
            logger.debug(f"Model name: {local_model_name}")

            openai_llm = OpenAILLM(
                model_name=local_model_name,
                openai_api_key=self._config.openai_api_key,
                message_stream=self._config.openai_streaming,
                stream_url=self._config.websocket_url,
                parameters=model_parameters,
                url=self._config.openai_url,
            )
            return openai_llm

        if local_engine_name == "anthropic":

            from elemental_agents.llm.llm_anthropic import AnthropicLLM

            local_model_name = (
                llm_parameters[1]
                if len(llm_parameters) > 1
                else self._config.anthropic_llm_model_name
            )

            logger.debug("Creating Anthropic LLM instance.")
            logger.debug(f"Model name: {local_model_name}")

            anthropic_llm = AnthropicLLM(
                model_name=local_model_name,
                message_stream=self._config.anthropic_streaming,
                stream_url=self._config.websocket_url,
                parameters=model_parameters,
                api_key=self._config.anthropic_api_key,
            )
            return anthropic_llm

        if local_engine_name == "custom":

            from elemental_agents.llm.llm_openai import OpenAILLM

            local_model_name = (
                llm_parameters[1]
                if len(llm_parameters) > 1
                else self._config.openai_llm_model_name
            )

            logger.debug("Creating Custom OpenAI LLM instance.")
            logger.debug(f"Model name: {local_model_name}")

            openai_llm = OpenAILLM(
                model_name=local_model_name,
                openai_api_key=self._config.custom_api_key,
                message_stream=self._config.custom_streaming,
                stream_url=self._config.websocket_url,
                parameters=model_parameters,
                url=self._config.custom_url,
            )
            return openai_llm

        if local_engine_name == "azure_openai":

            from elemental_agents.llm.llm_azure_openai import AzureOpenAILLM

            local_model_name = (
                llm_parameters[1]
                if len(llm_parameters) > 1
                else self._config.azure_openai_deployment_name
            )

            logger.debug("Creating Azure OpenAI LLM instance.")
            logger.debug(f"Deployment name: {local_model_name}")

            azure_openai_llm = AzureOpenAILLM(
                model_name=local_model_name,
                message_stream=self._config.azure_openai_streaming,
                stream_url=self._config.websocket_url,
                parameters=model_parameters,
                api_key=self._config.azure_openai_api_key,
                azure_endpoint=self._config.azure_openai_endpoint,
                api_version=self._config.azure_openai_api_version,
            )
            return azure_openai_llm

        if local_engine_name == "bedrock_anthropic":

            from elemental_agents.llm.llm_bedrock_anthropic import BedrockAnthropicLLM

            local_model_name = (
                llm_parameters[1]
                if len(llm_parameters) > 1
                else self._config.bedrock_anthropic_model_id
            )

            logger.debug("Creating Bedrock Anthropic LLM instance.")
            logger.debug(f"Model ID: {local_model_name}")

            bedrock_anthropic_llm = BedrockAnthropicLLM(
                model_name=local_model_name,
                message_stream=self._config.bedrock_anthropic_streaming,
                stream_url=self._config.websocket_url,
                parameters=model_parameters,
                aws_access_key_id=self._config.aws_access_key_id,
                aws_secret_access_key=self._config.aws_secret_access_key,
                aws_session_token=self._config.aws_session_token,
                region_name=self._config.aws_region,
            )
            return bedrock_anthropic_llm

        if local_engine_name == "google":

            from elemental_agents.llm.llm_gemini import GeminiLLM

            local_model_name = (
                llm_parameters[1]
                if len(llm_parameters) > 1
                else self._config.gemini_llm_model_name
            )

            logger.debug("Creating Gemini LLM instance.")
            logger.debug(f"Model name: {local_model_name}")

            gemini_llm = GeminiLLM(
                model_name=local_model_name,
                message_stream=self._config.gemini_streaming,
                stream_url=self._config.websocket_url,
                parameters=model_parameters,
                gemini_api_key=self._config.gemini_api_key
            )
            return gemini_llm


        if local_engine_name == "mock":

            from elemental_agents.llm.llm_mock import MockLLM

            logger.debug("Creating Mock LLM instance.")
            mock_llm = MockLLM(
                parameters=model_parameters,
            )
            return mock_llm

        logger.error(f"Unknown model name: {engine_name}")
        raise ValueError(f"Unknown model name: {engine_name}")


if __name__ == "__main__":
    from rich.console import Console

    from elemental_agents.llm.data_model import Message

    factory = LLMFactory()

    # Example with reasoning
    reasoning_params = ModelParameters(
        reasoning_effort="medium",  # For OpenAI
        thinking_enabled=True,  # For Anthropic/Ollama
        thinking_budget_tokens=1600,  # For Anthropic
    )

    model = factory.create(model_parameters=reasoning_params)
    msgs = [
        Message(role="system", content="You are helpful assistant."),
        Message(role="user", content="What is 10 + 23? Think step by step."),
    ]
    result = model.run(msgs)

    console = Console()
    console.print(result)
