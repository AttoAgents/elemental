"""
Abstraction of OpenAI module for the LLM class.
"""

from typing import Any, Dict, List, Optional

from loguru import logger
from openai import OpenAI
from openai.types.chat import ChatCompletion

from elemental_agents.llm.data_model import ModelParameters
from elemental_agents.llm.llm import LLM


class OpenAILLM(LLM):
    """
    OpenAILLM class that represents a language model inference type in the agent
    framework. This class is used to interact with the OpenAI API.
    """

    def __init__(
        self,
        model_name: str,
        message_stream: bool = False,
        stream_url: str = None,
        parameters: ModelParameters = ModelParameters(),
        openai_api_key: str = None,
        url: str = None,
        max_retries: int = 3,
    ) -> None:
        """
        Initialize the OpenAILLM object with the given parameters.

        :param model_name: The name of the model to use.
        :param message_stream: Whether to stream the messages.
        :param stream_url: The URL to stream the messages.
        :param parameters: The parameters for the model.
        :param openai_api_key: The OpenAI API key.
        :param url: The base URL for the OpenAI API.
        :param max_retries: Maximum number of retry attempts for API calls.
        """
        super().__init__(
            model_name, message_stream, stream_url, parameters, max_retries
        )

        self._client = OpenAI(api_key=openai_api_key, base_url=url)

    def _run_non_streaming(self, messages: List[Dict], stop_list: List[str]) -> str:
        """
        Run the model in non-streaming mode.

        :param messages: Serialized messages
        :param stop_list: List of stop words
        :return: Model response
        """
        output: ChatCompletion = self._client.chat.completions.create(
            model=self._model,
            messages=messages,  # type: ignore
            stream=False,
            temperature=self._temperature,
            max_completion_tokens=self._max_tokens,
            frequency_penalty=self._frequency_penalty,
            presence_penalty=self._presence_penalty,
            top_p=self._top_p,
            stop=stop_list,
        )

        logger.debug(f"Output: {output}")

        result = output.choices[0].message.content
        total_tokens = output.usage.total_tokens
        logger.debug(f"Total tokens used: {total_tokens}")

        return result

    async def _process_stream(self, messages: List[Dict], stop_list: List[str]) -> Any:
        """
        Process the stream from the model.

        :param messages: Serialized messages
        :param stop_list: List of stop words
        :return: Stream object from the model
        """
        return self._client.chat.completions.create(
            model=self._model,
            messages=messages,  # type: ignore
            stream=True,
            temperature=self._temperature,
            max_completion_tokens=self._max_tokens,
            frequency_penalty=self._frequency_penalty,
            presence_penalty=self._presence_penalty,
            top_p=self._top_p,
            stop=stop_list,
        )

    def _extract_content_from_chunk(self, chunk: Any) -> Optional[str]:
        """
        Extract content from a chunk in the stream.

        :param chunk: A chunk from the stream
        :return: The content from the chunk, or None if no content
        """
        return chunk.choices[0].delta.content
