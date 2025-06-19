"""
Abstraction of OpenAI module for the LLM class.
"""

from typing import Any, Dict, List, Optional

from loguru import logger
from openai import OpenAI
from openai.types.chat import ChatCompletion

from elemental_agents.llm_v.data_model import ModelParameters
from elemental_agents.llm_v.llm import LLM


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
        :param message_stream: Whether to stream messages.
        :param stream_url: The URL for streaming messages.
        :param parameters: Model parameters for the LLM.
        :param openai_api_key: API key for OpenAI.
        :param url: The base URL for the OpenAI API.
        :param max_retries: Maximum number of retries for API calls.
        """
        super().__init__(
            model_name, message_stream, stream_url, parameters, max_retries
        )

        self._client = OpenAI(api_key=openai_api_key, base_url=url)

    def _convert_message_for_openai(self, message: Dict) -> Dict:
        """
        Convert message format to OpenAI's expected format.

        :param message: Message in internal format
        :return: Message in OpenAI format
        """
        content = message.get("content", "")

        if isinstance(content, str):
            return message

        # Handle structured content
        openai_content = []

        for part in content:
            if isinstance(part, dict):
                part_type = part.get("type")

                if part_type == "text":
                    openai_content.append(
                        {"type": "text", "text": part.get("text", "")}
                    )
                elif part_type == "image":
                    # Convert ImageContent to OpenAI format
                    try:
                        if "base64_data" in part and part["base64_data"]:
                            # Use base64 data
                            media_type = part.get("media_type", "image/png")
                            openai_content.append(
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": (
                                            f"data:{media_type};base64,{part['base64_data']}"
                                        )
                                    },
                                }
                            )
                        elif "file_path" in part and part["file_path"]:
                            # Convert file to base64
                            import base64
                            import os

                            file_path = part["file_path"]
                            if os.path.exists(file_path):
                                with open(file_path, "rb") as image_file:
                                    base64_data = base64.b64encode(
                                        image_file.read()
                                    ).decode("utf-8")
                                media_type = part.get("media_type", "image/png")
                                openai_content.append(
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": (
                                                f"data:{media_type};base64,{base64_data}"
                                            )
                                        },
                                    }
                                )
                            else:
                                logger.warning(f"Image file not found: {file_path}")
                    except Exception as e:
                        logger.error(f"Error processing image: {e}")

        return {
            "role": message.get("role", "user"),
            "content": openai_content if openai_content else content,
        }

    def _has_image_content(self, message: Dict) -> bool:
        """
        Check if a message contains image content.

        :param message: The message to check.
        :return: True if the message contains images, False otherwise.
        """
        content = message.get("content", "")
        if isinstance(content, list):
            return any(
                isinstance(item, dict) and item.get("type") == "image"
                for item in content
            )
        return False

    def _run_non_streaming(self, messages: List[Dict], stop_list: List[str]) -> str:
        """
        Run the model in non-streaming mode.

        :param messages: List of messages to send to the model.
        :param stop_list: List of stop words to use in the model.
        :return: The model's response as a string.
        """
        # Convert messages to OpenAI format
        openai_messages = [self._convert_message_for_openai(msg) for msg in messages]

        # Prepare API call parameters
        api_params = {
            "model": self._model,
            "messages": openai_messages,
            "stream": False,
            "temperature": self._temperature,
            "max_completion_tokens": self._max_tokens,
            "frequency_penalty": self._frequency_penalty,
            "presence_penalty": self._presence_penalty,
            "top_p": self._top_p,
        }

        # Only add stop parameter if we have stop words
        if stop_list:
            api_params["stop"] = stop_list

        # Add reasoning effort for supported models
        if self._reasoning_effort and ("o1" in self._model or "o3" in self._model):
            api_params["reasoning_effort"] = self._reasoning_effort
            logger.debug(f"Using reasoning effort: {self._reasoning_effort}")

        output: ChatCompletion = self._client.chat.completions.create(**api_params) # type: ignore

        result = output.choices[0].message.content
        if output.usage:
            total_tokens = output.usage.total_tokens
            logger.debug(f"Total tokens used: {total_tokens}")

        return result

    async def _process_stream(self, messages: List[Dict], stop_list: List[str]) -> Any:
        """
        Process the stream from the model.
        """
        # Convert messages to OpenAI format
        openai_messages = [self._convert_message_for_openai(msg) for msg in messages]

        # Prepare API call parameters
        api_params = {
            "model": self._model,
            "messages": openai_messages,
            "stream": True,
            "temperature": self._temperature,
            "max_completion_tokens": self._max_tokens,
            "frequency_penalty": self._frequency_penalty,
            "presence_penalty": self._presence_penalty,
            "top_p": self._top_p,
        }

        if stop_list:
            api_params["stop"] = stop_list

        if self._reasoning_effort and ("o1" in self._model or "o3" in self._model):
            api_params["reasoning_effort"] = self._reasoning_effort

        return self._client.chat.completions.create(**api_params) # type: ignore

    def _extract_content_from_chunk(self, chunk: Any) -> Optional[str]:
        """
        Extract content from a chunk in the stream.
        """
        if chunk.choices and len(chunk.choices) > 0:
            delta = chunk.choices[0].delta
            return delta.content if hasattr(delta, "content") else None
        return None
