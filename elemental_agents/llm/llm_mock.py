"""
Mock version of the LLM class that represents a mocked language model in the framework.
"""

import asyncio
from typing import Any, Dict, List, Optional

from loguru import logger

from elemental_agents.llm.data_model import ModelParameters
from elemental_agents.llm.llm import LLM


class MockLLM(LLM):
    """
    MockLLM class that represents a mocked language model in the framework.
    """

    def __init__(
        self,
        message_stream: bool = False,
        stream_url: str = None,
        parameters: ModelParameters = ModelParameters(),
        max_retries: int = 3,
    ) -> None:
        """
        Initialize the MockLLM object.

        :param message_stream: Whether to stream the messages.
        :param stream_url: The URL to stream the messages.
        :param parameters: The parameters for the model.
        :param max_retries: Maximum number of retry attempts for API calls.
        """
        super().__init__(
            model_name="MockLLM",
            message_stream=message_stream,
            stream_url=stream_url,
            parameters=parameters,
            max_retries=max_retries,
        )

    def _run_non_streaming(self, messages: List[Dict], stop_list: List[str]) -> str:
        """
        Run the model in non-streaming mode.

        :param messages: Serialized messages
        :param stop_list: List of stop words
        :return: Model response
        """
        try:
            if len(messages) > 0:
                last_message = messages[-1]
                if last_message["role"] == "user":
                    result = f"Output for user message ({last_message['content']})"
                else:
                    result = f"Output for system message ({last_message['content']})"
            else:
                result = "No messages to process."
                logger.error("No messages have been provided to the model.")

            return result
        except Exception as e:
            logger.error(f"Error in MockLLM non-streaming call: {str(e)}")
            raise

    async def _process_stream(self, messages: List[Dict], stop_list: List[str]) -> Any:
        """
        Process the stream from the model.

        :param messages: Serialized messages
        :param stop_list: List of stop words
        :return: Stream object from the model
        """
        try:
            # Generate a mock response
            if len(messages) > 0:
                last_message = messages[-1]
                if last_message["role"] == "user":
                    full_response = (
                        f"Output for user message ({last_message['content']})"
                    )
                else:
                    full_response = (
                        f"Output for system message ({last_message['content']})"
                    )
            else:
                full_response = "No messages to process."
                logger.error("No messages have been provided to the model.")

            # Create a generator that simulates streaming by yielding one character at a time
            async def stream_generator() -> Any:
                for i, resp in enumerate(full_response):
                    # Create a mock chunk with the next character
                    chunk = {"content": resp, "index": i}
                    yield chunk
                    # Add a small delay to simulate real streaming
                    await asyncio.sleep(0)

            return stream_generator()
        except Exception as e:
            logger.error(f"Error in MockLLM streaming call: {str(e)}")
            raise

    def _extract_content_from_chunk(self, chunk: Any) -> Optional[str]:
        """
        Extract content from a chunk in the stream.

        :param chunk: A chunk from the stream
        :return: The content from the chunk, or None if no content
        """
        try:
            # Extract content from the mock chunk
            if chunk and "content" in chunk:
                return chunk["content"]
            return None
        except Exception as e:
            logger.error(f"Error extracting content from MockLLM chunk: {str(e)}")
            return None
