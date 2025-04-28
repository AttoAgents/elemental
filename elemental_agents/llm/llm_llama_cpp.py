"""
Class to use Llama CPP for language model inference in the agent framework.
"""

from typing import Any, Dict, List, Optional

from llama_cpp import Llama
from llama_cpp.llama_types import CreateChatCompletionResponse
from loguru import logger

from elemental_agents.llm.data_model import ModelParameters
from elemental_agents.llm.llm import LLM


class LlamaCppLLM(LLM):
    """
    Llama CPP language model inference class.
    """

    def __init__(
        self,
        model_name: str,
        message_stream: bool = False,
        stream_url: str = None,
        parameters: ModelParameters = ModelParameters(),
        max_retries: int = 3,
        n_ctx: int = 2048,
        n_threads: Optional[int] = None,
    ) -> None:
        """
        Initialize the LlamaCppLLM object with the given parameters.

        :param model_name: The name of the model to use.
        :param message_stream: Whether to stream the messages.
        :param stream_url: The URL to stream the messages.
        :param parameters: The parameters for the model.
        :param max_retries: Maximum number of retry attempts for API calls.
        :param n_ctx: Size of the context window for the model.
        :param n_threads: Number of threads to use for computation (None = auto).
        """
        super().__init__(
            model_name, message_stream, stream_url, parameters, max_retries
        )

        self.llm_model = Llama(
            model_path=model_name,
            chat_format="chatml",
            n_ctx=n_ctx,
            n_threads=n_threads,
        )

    def _run_non_streaming(self, messages: List[Dict], stop_list: List[str]) -> str:
        """
        Run the model in non-streaming mode.

        :param messages: Serialized messages
        :param stop_list: List of stop words
        :return: Model response
        """
        try:
            response: CreateChatCompletionResponse = (
                self.llm_model.create_chat_completion(
                    messages=messages,  # type: ignore
                    temperature=self._temperature,
                    stop=stop_list,
                    max_tokens=self._max_tokens,
                    presence_penalty=self._presence_penalty,
                    frequency_penalty=self._frequency_penalty,
                    top_p=self._top_p,
                    stream=False,
                )
            )

            output = response["choices"][0]["message"]["content"]
            return output
        except (ValueError, RuntimeError) as e:
            logger.error(f"Error in Llama CPP non-streaming call: {str(e)}")
            raise

    async def _process_stream(self, messages: List[Dict], stop_list: List[str]) -> Any:
        """
        Process the stream from the model.

        :param messages: Serialized messages
        :param stop_list: List of stop words
        :return: Stream object from the model
        """
        try:
            # Create a generator that yields chunks from llama-cpp
            def stream_generator() -> Any:
                stream = self.llm_model.create_chat_completion(
                    messages=messages,  # type: ignore
                    temperature=self._temperature,
                    stop=stop_list,
                    max_tokens=self._max_tokens,
                    presence_penalty=self._presence_penalty,
                    frequency_penalty=self._frequency_penalty,
                    top_p=self._top_p,
                    stream=True,
                )
                yield from stream

            return stream_generator()
        except (ValueError, RuntimeError) as e:
            logger.error(f"Error in Llama CPP streaming call: {str(e)}")
            raise

    def _extract_content_from_chunk(self, chunk: Any) -> Optional[str]:
        """
        Extract content from a chunk in the stream.

        :param chunk: A chunk from the stream
        :return: The content from the chunk, or None if no content
        """
        try:
            # Extract content from the streaming chunk
            if chunk and "choices" in chunk and len(chunk["choices"]) > 0:
                delta = chunk["choices"][0].get("delta", {})
                if "content" in delta and delta["content"]:
                    return delta["content"]
            return None
        except (KeyError, TypeError) as e:
            logger.error(f"Error extracting content from Llama CPP chunk: {str(e)}")
            return None
