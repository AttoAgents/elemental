"""
Abstraction of Ollama module for the LLM class.
"""

from typing import Any, Dict, List, Optional

from ollama import Client, Options

from elemental_agents.llm.data_model import ModelParameters
from elemental_agents.llm.llm import LLM


class OllamaLLM(LLM):
    """
    OllamaLLM class that represents a language model inference type in the agent
    framework. This class is used to interact with the Ollama API.
    """

    def __init__(
        self,
        model_name: str,
        message_stream: bool = False,
        stream_url: str = None,
        parameters: ModelParameters = ModelParameters(),
        url: str = None,
        max_retries: int = 3,
    ) -> None:
        """
        Initialize the OllamaLLM object with the given parameters.

        :param model_name: The name of the model to use.
        :param message_stream: Whether to stream the messages.
        :param stream_url: The URL to stream the messages.
        :param parameters: The parameters for the model.
        :param url: The base URL for the Ollama API.
        :param max_retries: Maximum number of retry attempts for API calls.
        """
        super().__init__(
            model_name, message_stream, stream_url, parameters, max_retries
        )

        self._client = Client(host=url)

    def _prepare_options(self, stop_list: List[str]) -> Options:
        """
        Prepare options for the Ollama model.

        :param stop_list: List of stop words
        :return: Options for the model
        """
        options: Options = {}
        options["temperature"] = self._temperature
        options["num_predict"] = self._max_tokens
        options["stop"] = stop_list
        options["frequency_penalty"] = self._frequency_penalty
        options["presence_penalty"] = self._presence_penalty
        options["top_p"] = self._top_p
        return options

    def _run_non_streaming(self, messages: List[Dict], stop_list: List[str]) -> str:
        """
        Run the model in non-streaming mode.

        :param messages: Serialized messages
        :param stop_list: List of stop words
        :return: Model response
        """
        options = self._prepare_options(stop_list)

        output = self._client.chat(
            model=self._model,
            messages=messages,  # type: ignore
            stream=False,
            options=options,
        )

        result = output["message"]["content"]  # type: ignore
        return result

    async def _process_stream(self, messages: List[Dict], stop_list: List[str]) -> Any:
        """
        Process the stream from the model.

        :param messages: Serialized messages
        :param stop_list: List of stop words
        :return: Stream object from the model
        """
        options = self._prepare_options(stop_list)

        return self._client.chat(
            model=self._model,
            messages=messages,  # type: ignore
            stream=True,
            options=options,
        )

    def _extract_content_from_chunk(self, chunk: Any) -> Optional[str]:
        """
        Extract content from a chunk in the stream.

        :param chunk: A chunk from the stream
        :return: The content from the chunk, or None if no content
        """
        return chunk["message"]["content"]
