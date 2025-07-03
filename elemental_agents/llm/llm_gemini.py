"""
Abstraction of Google Gemini module for the LLM class.
"""

import base64
import os
from typing import Any, Dict, List, Optional

from google import genai
from google.genai import types
from loguru import logger

from elemental_agents.llm.data_model import ModelParameters
from elemental_agents.llm.llm import LLM


class GeminiLLM(LLM):
    """
    GeminiLLM class that represents a language model inference type in the agent
    framework. This class is used to interact with the Google Gemini API.
    """

    def __init__(
        self,
        model_name: str,
        message_stream: bool = False,
        stream_url: str = None,
        parameters: ModelParameters = ModelParameters(),
        gemini_api_key: str = None,
        max_retries: int = 3,
    ) -> None:
        """
        Initialize the GeminiLLM object with the given parameters.

        :param model_name: The name of the model to use (e.g., 'gemini-pro', 'gemini-pro-vision').
        :param message_stream: Whether to stream messages.
        :param stream_url: The URL for streaming messages (not used for Gemini).
        :param parameters: Model parameters for the LLM.
        :param gemini_api_key: API key for Google Gemini.
        :param max_retries: Maximum number of retries for API calls.
        """
        super().__init__(
            model_name, message_stream, stream_url, parameters, max_retries
        )

        # Initialize the client
        self._client = genai.Client(api_key=gemini_api_key)

        # Set up generation config
        self._generation_config = types.GenerateContentConfig(
            temperature=self._temperature,
            max_output_tokens=self._max_tokens,
            top_p=self._top_p,
            stop_sequences=parameters.stop if parameters.stop else [],
        )

    def _convert_message_for_gemini(self, message: Dict) -> Dict:
        """
        Convert message format to Gemini's expected format.

        :param message: Message in internal format
        :return: Message in Gemini format
        """
        role = message.get("role", "user")
        content = message.get("content", "")

        # Map roles to Gemini format
        gemini_role = "user" if role in ["user", "system"] else "model"

        if isinstance(content, str):
            return {"role": gemini_role, "parts": [content]}

        # Handle structured content (multimodal)
        parts = []

        for part in content:
            if isinstance(part, dict):
                part_type = part.get("type")

                if part_type == "text":
                    parts.append(part.get("text", ""))
                elif part_type == "image":
                    try:
                        if "base64_data" in part and part["base64_data"]:
                            # Use base64 data directly
                            media_type = part.get("media_type", "image/png")
                            image_data = base64.b64decode(part["base64_data"])
                            parts.append({"mime_type": media_type, "data": image_data})
                        elif "file_path" in part and part["file_path"]:
                            # Read file and convert to bytes
                            file_path = part["file_path"]
                            if os.path.exists(file_path):
                                with open(file_path, "rb") as image_file:
                                    image_data = image_file.read()
                                media_type = part.get("media_type", "image/png")
                                parts.append(
                                    {"mime_type": media_type, "data": image_data}
                                )
                            else:
                                logger.warning(f"Image file not found: {file_path}")
                    except Exception as e:
                        logger.error(f"Error processing image: {e}")

        return {"role": gemini_role, "parts": parts if parts else [content]}

    def _convert_messages_to_gemini_history(
            self,
            messages: List[Dict]
        ) -> tuple[List[Dict], List[Dict]]:
        """
        Convert messages to Gemini chat history format.

        :param messages: List of messages in internal format
        :return: List of messages in Gemini format
        """
        gemini_messages = []
        system_instruction = None

        for message in messages:
            role = message.get("role", "user")

            # Handle system messages separately
            if role == "system":
                system_instruction = message.get("content", "")
                continue

            converted_message = self._convert_message_for_gemini(message)
            gemini_messages.append(converted_message)

        return gemini_messages, system_instruction

    def _run_non_streaming(self, messages: List[Dict], stop_list: List[str]) -> str:
        """
        Run the model in non-streaming mode.

        :param messages: List of messages to send to the model.
        :param stop_list: List of stop words to use in the model (not directly supported by Gemini).
        :return: The model's response as a string.
        """
        try:
            gemini_messages, system_instruction = (
                self._convert_messages_to_gemini_history(messages)
            )

            # Update generation config with stop sequences if provided and system instruction
            generation_config = self._generation_config
            if stop_list:
                generation_config = types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=self._temperature,
                    max_output_tokens=self._max_tokens,
                    top_p=self._top_p,
                    stop_sequences=stop_list[
                        :5
                    ],  # Gemini supports up to 5 stop sequences
                )
            else:
                generation_config = types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=self._temperature,
                    max_output_tokens=self._max_tokens,
                    top_p=self._top_p,
                )

            client = self._client

            if len(gemini_messages) == 1:
                # Single message generation
                response = client.models.generate_content(
                    model=self._model,
                    contents=gemini_messages[0]["parts"],
                    config=generation_config
                )
            else:
                # Multi-turn conversation
                chat = client.chats.create(
                    model=self._model,
                    config=generation_config,
                    history=gemini_messages[:-1])
                response = chat.send_message(
                    gemini_messages[-1]["parts"]
                )

            # Extract text from response
            if response.candidates and len(response.candidates) > 0:
                candidate = response.candidates[0]
                if candidate.content and candidate.content.parts:
                    result = "".join(
                        [
                            part.text
                            for part in candidate.content.parts
                            if hasattr(part, "text")
                        ]
                    )

                    # Log usage information if available
                    if hasattr(response, "usage_metadata") and response.usage_metadata:
                        total_tokens = (
                            response.usage_metadata.prompt_token_count
                            + response.usage_metadata.candidates_token_count
                        )
                        logger.debug(f"Total tokens used: {total_tokens}")

                    return result

            logger.warning("No valid response generated")
            return ""

        except Exception as e:
            logger.error(f"Error in Gemini API call: {e}")
            raise

    async def _process_stream(self, messages: List[Dict], stop_list: List[str]) -> Any:
        """
        Process the stream from the model.

        :param messages: List of messages to send to the model.
        :param stop_list: List of stop words to use in the model.
        :return: Async generator for streaming response.
        """
        try:
            gemini_messages, system_instruction = (
                self._convert_messages_to_gemini_history(messages)
            )

            # Update generation config with stop sequences if provided
            generation_config = self._generation_config
            if stop_list:
                generation_config = types.GenerateContentConfig(
                    temperature=self._temperature,
                    system_instruction=system_instruction,
                    max_output_tokens=self._max_tokens,
                    top_p=self._top_p,
                    stop_sequences=stop_list[
                        :5
                    ],  # Gemini supports up to 5 stop sequences
                )

            client = self._client

            if len(gemini_messages) == 1:
                # Single message generation with streaming
                return client.models.generate_content_stream(
                    model=self._model,
                    contents=gemini_messages[0]["parts"],
                    config=generation_config,
                )
            else:
                # Multi-turn conversation with streaming
                chat = client.chats.create(
                    model=self._model,
                    config=generation_config,
                    history=gemini_messages[:-1])
                response = chat.send_message_stream(
                    gemini_messages[-1]["parts"]
                )

            # Extract text from response
            if response.candidates and len(response.candidates) > 0:
                candidate = response.candidates[0]
                if candidate.content and candidate.content.parts:
                    result = "".join(
                        [
                            part.text
                            for part in candidate.content.parts
                            if hasattr(part, "text")
                        ]
                    )

                    # Log usage information if available
                    if hasattr(response, "usage_metadata") and response.usage_metadata:
                        total_tokens = (
                            response.usage_metadata.prompt_token_count
                            + response.usage_metadata.candidates_token_count
                        )
                        logger.debug(f"Total tokens used: {total_tokens}")

                    return result

            logger.warning("No valid response generated")
            return ""

        except Exception as e:
            logger.error(f"Error in Gemini streaming API call: {e}")
            raise

    def _extract_content_from_chunk(self, chunk: Any) -> Optional[str]:
        """
        Extract content from a chunk in the stream.

        :param chunk: Chunk from Gemini streaming response.
        :return: Extracted text content or None.
        """
        try:
            if hasattr(chunk, "candidates") and chunk.candidates:
                candidate = chunk.candidates[0]
                if hasattr(candidate, "content") and candidate.content:
                    if hasattr(candidate.content, "parts") and candidate.content.parts:
                        # Extract text from all parts
                        text_parts = []
                        for part in candidate.content.parts:
                            if hasattr(part, "text") and part.text:
                                text_parts.append(part.text)
                        return "".join(text_parts) if text_parts else None
            return None
        except Exception as e:
            logger.error(f"Error extracting content from chunk: {e}")
            return None
