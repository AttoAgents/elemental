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
        :param message_stream: Whether to stream messages.
        :param stream_url: The URL for streaming messages.
        :param parameters: Model parameters for the LLM.
        :param api_key: API key for Anthropic.
        :param max_retries: Maximum number of retries for API calls.
        """
        super().__init__(
            model_name, message_stream, stream_url, parameters, max_retries
        )

        self._client = Anthropic(api_key=api_key)

    def _extract_system_message(self, messages: List[Dict]) -> tuple:
        """
        Extract the system message from the list of messages.

        :param messages: List of messages in internal format
        :return: Tuple containing the system message and the remaining messages
        """
        if not messages:
            return "", []

        system_message = ""
        remaining_messages = []

        if messages[0].get("role") == "system":
            content = messages[0].get("content", "")
            if isinstance(content, str):
                system_message = content
            else:
                # Extract only text parts from structured content
                text_parts = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
                system_message = " ".join(text_parts)
            remaining_messages = messages[1:]
        else:
            remaining_messages = messages

        return system_message, remaining_messages

    def _convert_message_for_anthropic(self, message: Dict) -> Dict:
        """
        Convert message format to Anthropic's expected format.

        :param message: Message in internal format
        :return: Message in Anthropic format
        """
        content = message.get("content", "")

        if isinstance(content, str):
            return message

        # Handle structured content
        anthropic_content = []

        for part in content:
            if isinstance(part, dict):
                part_type = part.get("type")

                if part_type == "text":
                    anthropic_content.append(
                        {"type": "text", "text": part.get("text", "")}
                    )
                elif part_type == "image":
                    # Convert ImageContent to Anthropic format (base64 only)
                    try:
                        base64_data = None
                        media_type = part.get("media_type", "image/png")

                        if "base64_data" in part and part["base64_data"]:
                            base64_data = part["base64_data"]
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
                            else:
                                logger.warning(f"Image file not found: {file_path}")
                                continue

                        if base64_data:
                            anthropic_content.append(
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": media_type,
                                        "data": base64_data,
                                    },
                                }
                            )
                    except Exception as e:
                        logger.error(f"Error processing image for Anthropic: {e}")

        return {
            "role": message.get("role", "user"),
            "content": anthropic_content if anthropic_content else content,
        }

    def _run_non_streaming(self, messages: List[Dict], stop_list: List[str]) -> str:
        """
        Run the model in non-streaming mode.
        """
        system_message, remaining_messages = self._extract_system_message(messages)

        # Convert messages to Anthropic format
        anthropic_messages = [
            self._convert_message_for_anthropic(msg) for msg in remaining_messages
        ]

        # Prepare API call parameters
        api_params = {
            "max_tokens": self._max_tokens,
            "messages": anthropic_messages,
            "model": self._model,
            "stream": False,
            "temperature": self._temperature,
            "top_p": self._top_p,
        }

        if stop_list:
            api_params["stop_sequences"] = stop_list

        if system_message:
            api_params["system"] = system_message

        # Add thinking parameters for supported models
        if self._thinking_enabled and (
            "claude-3-5" in self._model or "claude-3-7" in self._model
        ):
            thinking_config: Dict[str, Any] = {"type": "enabled"}
            if self._thinking_budget_tokens:
                thinking_config["budget_tokens"] = self._thinking_budget_tokens
            api_params["thinking"] = thinking_config
            logger.debug(f"Using thinking mode with config: {thinking_config}")

        output = self._client.messages.create(**api_params)  # type: ignore

        # Handle thinking content
        if hasattr(output, "content") and isinstance(output.content, list):
            result_parts = []
            for block in output.content:
                if hasattr(block, "type"):
                    if block.type == "thinking":
                        logger.debug(f"Thinking: {block.thinking}")
                    elif block.type == "text":
                        result_parts.append(block.text)
            return " ".join(result_parts)
        else:
            return str(output.content)

    async def _process_stream(self, messages: List[Dict], stop_list: List[str]) -> Any:
        """
        Process the stream from the model.
        """
        system_message, remaining_messages = self._extract_system_message(messages)
        anthropic_messages = [
            self._convert_message_for_anthropic(msg) for msg in remaining_messages
        ]

        api_params = {
            "max_tokens": self._max_tokens,
            "messages": anthropic_messages,
            "model": self._model,
            "temperature": self._temperature,
            "top_p": self._top_p,
            "stream": True,
        }

        if stop_list:
            api_params["stop_sequences"] = stop_list
        if system_message:
            api_params["system"] = system_message

        if self._thinking_enabled and (
            "claude-3-5" in self._model or "claude-3-7" in self._model
        ):
            thinking_config: Dict[str, Any] = {"type": "enabled"}
            if self._thinking_budget_tokens:
                thinking_config["budget_tokens"] = self._thinking_budget_tokens
            api_params["thinking"] = thinking_config

        return self._client.messages.create(**api_params)  # type: ignore

    def _extract_content_from_chunk(self, chunk: Any) -> Optional[str]:
        """
        Extract content from a chunk in the stream.
        """
        if chunk and hasattr(chunk, "type"):
            if chunk.type == "content_block_delta":
                if hasattr(chunk, "delta") and hasattr(chunk.delta, "text"):
                    return chunk.delta.text
            elif chunk.type == "content_block_start":
                if hasattr(chunk, "content_block") and hasattr(
                    chunk.content_block, "text"
                ):
                    return chunk.content_block.text
        return None
