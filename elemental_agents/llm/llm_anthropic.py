"""
Abstraction of Anthropic module for the LLM class.
"""

from typing import Any, Dict, List, Optional

from anthropic import Anthropic
from loguru import logger

from elemental_agents.llm.data_model import ModelParameters
from elemental_agents.llm.llm import LLM


class AnthropicLLM(LLM):
    """
    AnthropicLLM class that represents a language model inference type in the
    agent framework. This class is used to interact with the Anthropic API.
    """

    def __init__(
        self,
        model_name: str,
        message_stream: bool = False,
        stream_url: str = None,
        parameters: ModelParameters = ModelParameters(),
        api_key: str = None,
        max_retries: int = 3,
    ) -> None:
        """
        Initialize the AnthropicLLM object with the given parameters.

        :param model_name: The name of the model to use.
        :param message_stream: Whether to stream the messages.
        :param stream_url: The URL to stream the messages.
        :param parameters: The parameters for the model LLM.
        :param api_key: The Anthropic API key.
        :param max_retries: Maximum number of retry attempts for API calls.
        """
        super().__init__(
            model_name, message_stream, stream_url, parameters, max_retries
        )

        self._client = Anthropic(api_key=api_key)

    def _extract_system_message(self, messages: List[Dict]) -> tuple:
        """
        Extract the system message from the list of messages.

        :param messages: The list of messages.
        :return: A tuple containing the system message and the remaining messages.
        """
        if not messages:
            return "", []

        system_message = messages[0]["content"]
        remaining_messages = messages[1:]
        return system_message, remaining_messages

    def _run_non_streaming(self, messages: List[Dict], stop_list: List[str]) -> str:
        """
        Run the model in non-streaming mode.

        :param messages: Serialized messages
        :param stop_list: List of stop words
        :return: Model response
        """
        system_message, remaining_messages = self._extract_system_message(messages)

        output = self._client.messages.create(
            max_tokens=self._max_tokens,
            messages=remaining_messages,
            model=self._model,
            stop_sequences=stop_list,
            stream=False,
            temperature=self._temperature,
            top_p=self._top_p,
            system=system_message,
        )

        logger.debug(f"Output: {output}")
        return str(output.content)

    async def _process_stream(self, messages: List[Dict], stop_list: List[str]) -> Any:
        """
        Process the stream from the model.

        :param messages: Serialized messages
        :param stop_list: List of stop words
        :return: Stream object from the model
        """
        system_message, remaining_messages = self._extract_system_message(messages)

        return self._client.messages.create(
            max_tokens=self._max_tokens,
            messages=remaining_messages,
            model=self._model,
            stop_sequences=stop_list,
            temperature=self._temperature,
            top_p=self._top_p,
            stream=True,
            system=system_message,
        )

    def _extract_content_from_chunk(self, chunk: Any) -> Optional[str]:
        """
        Extract content from a chunk in the stream.

        :param chunk: A chunk from the stream
        :return: The content from the chunk, or None if no content
        """
        if chunk and (chunk.type == "content_block_delta"):
            return chunk.delta.text
        return None
