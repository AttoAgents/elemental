"""
Language model factory class to create LLM instances.
"""

import os
from typing import Optional

from loguru import logger

from elemental_agents.llm.data_model import ModelParameters
from elemental_agents.llm.llm import LLM
from elemental_agents.llm.llm_anthropic import AnthropicLLM
from elemental_agents.llm.llm_llama_cpp import LlamaCppLLM
from elemental_agents.llm.llm_mock import MockLLM
from elemental_agents.llm.llm_ollama import OllamaLLM
from elemental_agents.llm.llm_openai import OpenAILLM
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
        self, engine_name: str = None,
        model_parameters: Optional[ModelParameters] = ModelParameters()
    ) -> LLM:
        """
        Create an LLM instance based on the engine name. If the engine name is
        not provided, the default engine is used that is specified in the
        configuration file.

        :param engine_name: The name of the engine to use.
        :param model_parameters: The parameters for the LLM instance.
        :return: An instance of the LLM class.
        """

        llm_parameters = []

        if engine_name is None:
            local_engine_name = self._config.default_engine
        else:
            llm_parameters = engine_name.split("|")
            local_engine_name = llm_parameters[0]

        if local_engine_name == "ollama":

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

        if local_engine_name == "llama_cpp":

            local_model_name = (
                llm_parameters[1]
                if len(llm_parameters) > 1
                else self._config.llama_llm_model_name
            )

            logger.debug("Creating LlamaCpp LLM instance.")
            logger.debug(f"Model name: {local_model_name}")

            model_full_path = os.path.join(
                self._config.llama_cpp_model_directory, local_model_name
            )

            llama_cpp_llm = LlamaCppLLM(
                model_name=model_full_path,
                message_stream=self._config.llama_cpp_streaming,
                stream_url=self._config.websocket_url,
                parameters=model_parameters,
            )
            return llama_cpp_llm

        if local_engine_name == "custom":

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

        if local_engine_name == "mock":

            logger.debug("Creating Mock LLM instance.")
            mock_llm = MockLLM()
            return mock_llm

        logger.error(f"Unknown model name: {engine_name}")
        raise ValueError(f"Unknown model name: {engine_name}")


if __name__ == "__main__":

    from rich.console import Console

    from .data_model import Message

    factory = LLMFactory()
    model = factory.create()
    msgs = [
        Message(role="system", content="You are helpful assistant."),
        Message(role="user", content="The sky is blue"),
    ]
    result = model.run(msgs)

    console = Console()
    console.print(result)
