"""
Abstraction of Ollama module for the LLM class.
"""

from typing import Any, Dict, List, Optional

from loguru import logger
from ollama import Client, Options

from elemental_agents.llm_v.data_model import ModelParameters
from elemental_agents.llm_v.llm import LLM


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
        :param message_stream: Whether to stream messages.
        :param stream_url: The URL for streaming messages.
        :param parameters: Model parameters for the LLM.
        :param url: The base URL for the Ollama API.
        :param max_retries: Maximum number of retries for API calls.
        """
        super().__init__(
            model_name, message_stream, stream_url, parameters, max_retries
        )

        self._client = Client(host=url)

    def _prepare_options(self, stop_list: List[str]) -> Options:
        """
        Prepare options for the Ollama model.

        :param stop_list: List of stop sequences for the model.
        :return: Options object with the model parameters.
        """
        options: Options = {}
        options["temperature"] = self._temperature
        options["num_predict"] = self._max_tokens
        if stop_list:
            options["stop"] = stop_list
        options["frequency_penalty"] = self._frequency_penalty
        options["presence_penalty"] = self._presence_penalty
        options["top_p"] = self._top_p
        return options

    def _convert_message_for_ollama(self, message: Dict) -> Dict:
        """
        Convert message format to Ollama's expected format.
        """
        content = message.get("content", "")
        role = message.get("role", "user")

        if isinstance(content, str):
            return {"role": role, "content": content}

        # Handle structured content
        text_parts = []
        images = []

        for part in content:
            if isinstance(part, dict):
                part_type = part.get("type")

                if part_type == "text":
                    text_parts.append(part.get("text", ""))
                elif part_type == "image":
                    # Ollama prefers file paths when possible, falls back to base64
                    try:
                        if "file_path" in part and part["file_path"]:
                            import os

                            file_path = part["file_path"]
                            if os.path.exists(file_path):
                                images.append(file_path)
                                logger.debug(f"Using file path for Ollama: {file_path}")
                            else:
                                logger.warning(f"Image file not found: {file_path}")
                        elif "base64_data" in part and part["base64_data"]:
                            # Use base64 data
                            images.append(part["base64_data"])
                            logger.debug("Using base64 data for Ollama")
                    except Exception as e:
                        logger.error(f"Error processing image for Ollama: {e}")

        # Combine text parts
        combined_text = " ".join(text_parts) if text_parts else content

        # Create Ollama-formatted message
        result = {"role": role, "content": combined_text}
        if images:
            result["images"] = images

        return result

    def _run_non_streaming(self, messages: List[Dict], stop_list: List[str]) -> str:
        """
        Run the model in non-streaming mode.

        :param messages: List of messages to send to the model.
        :param stop_list: List of stop sequences for the model.
        :return: The model's response as a string.
        """
        options = self._prepare_options(stop_list)

        # Convert messages to Ollama format
        ollama_messages = [self._convert_message_for_ollama(msg) for msg in messages]

        # Prepare API call parameters
        api_params = {
            "model": self._model,
            "messages": ollama_messages,
            "stream": False,
            "options": options,
        }

        # Add thinking parameter for supported models
        if self._thinking_enabled and (
            "deepseek-r1" in self._model or "qwen2.5" in self._model
        ):
            api_params["think"] = True
            logger.debug("Using thinking mode")

        output = self._client.chat(**api_params)  # type: ignore

        # Handle thinking content
        if isinstance(output, dict) and "message" in output:
            message = output["message"]
            if "thinking" in message and message["thinking"]:
                logger.debug(f"Thinking: {message['thinking']}")
            result = message.get("content", "")
        else:
            result = str(output)

        return result

    async def _process_stream(self, messages: List[Dict], stop_list: List[str]) -> Any:
        """
        Process the stream from the model.
        """
        options = self._prepare_options(stop_list)
        ollama_messages = [self._convert_message_for_ollama(msg) for msg in messages]

        api_params = {
            "model": self._model,
            "messages": ollama_messages,
            "stream": True,
            "options": options,
        }

        if self._thinking_enabled and (
            "deepseek-r1" in self._model or "qwen2.5" in self._model
        ):
            api_params["think"] = True

        return self._client.chat(**api_params)  # type: ignore

    def _extract_content_from_chunk(self, chunk: Any) -> Optional[str]:
        """
        Extract content from a chunk in the stream.
        """
        if isinstance(chunk, dict):
            if "message" in chunk and "content" in chunk["message"]:
                return chunk["message"]["content"]
        return None
