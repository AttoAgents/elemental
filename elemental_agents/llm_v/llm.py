"""
Abstract class for the Language Model (LLM) in the Elemental framework.
"""

import asyncio
import io
import json
import os
from abc import ABC, abstractmethod
from base64 import b64encode
from typing import Any, Callable, Dict, List, Optional, Union

import requests.exceptions
import socketio
from loguru import logger
from PIL import Image
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from elemental_agents.llm_v.data_model import Message, ModelParameters


class LLM(ABC):
    """
    Interface for the Language Model (LLM) in the Atomic framework.
    """

    def __init__(
        self,
        model_name: str,
        message_stream: bool = False,
        stream_url: str = None,
        parameters: ModelParameters = ModelParameters(),
        max_retries: int = 3,
    ) -> None:
        """
        Initialize the LLM object.

        :param model_name: The name of the model to use.
        :param message_stream: Whether to stream the messages.
        :param stream_url: The URL to stream the messages.
        :param parameters: The parameters for the model.
        :param max_retries: Maximum number of retry attempts for API calls.
        """
        self._model = model_name
        self._stream = message_stream
        self._stream_url = stream_url
        self._sio = None
        self._max_retries = max_retries

        self._max_tokens = parameters.max_tokens
        self._temperature = parameters.temperature
        self._frequency_penalty = parameters.frequency_penalty
        self._presence_penalty = parameters.presence_penalty
        self._top_p = parameters.top_p

        # Reasoning parameters
        self._reasoning_effort = parameters.reasoning_effort
        self._thinking_enabled = parameters.thinking_enabled
        self._thinking_budget_tokens = parameters.thinking_budget_tokens

        if parameters.stop is None:
            self._stop = []
        else:
            self._stop = parameters.stop

    def set_temperature(self, temperature: float) -> None:
        """
        Set the temperature for the model.

        :param temperature: The temperature to set.
        """
        self._temperature = temperature

    def _encode_image(self, image_path: str) -> str:
        """
        Encode an image file to base64 string.

        :param image_path: Path to the image file
        :return: Base64 encoded string
        """
        with open(image_path, "rb") as image_file:
            return b64encode(image_file.read()).decode("utf-8")

    def _get_image_data(self, image: Union[str, bytes, Image.Image]) -> Dict:
        """
        Process different types of image inputs and convert to the format required by API.

        :param image: Can be a URL, file path, bytes, or PIL Image
        :return: Dictionary with image data in the format required by API
        """
        # If image is a URL
        if isinstance(image, str) and (
            image.startswith("http://") or image.startswith("https://")
        ):
            return {"url": image}

        # If image is a file path
        elif isinstance(image, str) and os.path.isfile(image):
            base64_image = self._encode_image(image)
            return {"base64": base64_image}

        # If image is bytes
        elif isinstance(image, bytes):
            base64_image = b64encode(image).decode("utf-8")
            return {"base64": base64_image}

        # If image is a PIL Image
        elif isinstance(image, Image.Image):
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            base64_image = b64encode(buffer.getvalue()).decode("utf-8")
            return {"base64": base64_image}

        else:
            raise ValueError(
                "Unsupported image format. Please provide a URL, file path, bytes, or PIL Image."
            )

    def _has_image_content(self, message: Dict) -> bool:
        """
        Check if a message contains image content.

        :param message: The message to check
        :return: True if the message contains image content, False otherwise
        """
        content = message.get("content", "")
        if isinstance(content, list):
            return any(
                item.get("type") == "image" or item.get("type") == "image_url"
                for item in content
            )
        return False

    def _get_retry_decorator(self) -> Callable:
        """
        Returns a configured retry decorator using the instance's max_retries value.
        """
        return retry(
            stop=stop_after_attempt(self._max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            retry=retry_if_exception_type(
                (
                    requests.exceptions.RequestException,
                    ConnectionError,
                    TimeoutError,
                    Exception,
                )
            ),
            reraise=True,
        )

    def run(self, messages: List[Message], stop_word: str | None = None) -> str:
        """
        Run the LLM model with the given messages.
        Will retry on network-related exceptions.

        :param messages: The list of messages to process.
        :param stop_word: The stop word to use for the model.
        :return: The result of the model processing the messages. Only the raw
            assistant response is returned.
        """

        # Apply the retry decorator dynamically
        @self._get_retry_decorator()
        def _run_with_retry(
            messages: List[Message], stop_word: str | None = None
        ) -> str:
            try:
                # Serialize the messages
                msg = [m.model_dump() for m in messages]
                if len(msg) == 0:
                    logger.error("No messages have been provided to the model.")
                    return "No messages to process."

                # Check if any messages contain images
                has_images = any(message.is_multimodal() for message in messages)
                if has_images:
                    logger.debug("Processing multimodal messages with images")

                # Prepare stop words
                stop_list = self._stop.copy() if self._stop else []
                if stop_word and stop_word not in stop_list:
                    stop_list.append(stop_word)
                logger.debug(f"Stop words: {stop_list}")

                if not self._stream:
                    # Non-streaming mode
                    return self._run_non_streaming(msg, stop_list)
                else:
                    # Streaming mode
                    queue: asyncio.Queue[str] = asyncio.Queue()
                    asyncio.run(self._run_streaming(messages, queue, stop_list))
                    return asyncio.run(queue.get())
            except Exception as e:
                logger.error(f"Error in LLM run: {str(e)}")
                raise

        # Call the decorated function
        return _run_with_retry(messages, stop_word)

    @abstractmethod
    def _run_non_streaming(self, messages: List[Dict], stop_list: List[str]) -> str:
        """
        Run the model in non-streaming mode.

        :param messages: Serialized messages
        :param stop_list: List of stop words
        :return: Model response
        """

    @abstractmethod
    async def _process_stream(self, messages: List[Dict], stop_list: List[str]) -> Any:
        """
        Process the stream from the model.

        :param messages: Serialized messages
        :param stop_list: List of stop words
        :return: Stream object from the model
        """

    @abstractmethod
    def _extract_content_from_chunk(self, chunk: Any) -> Optional[str]:
        """
        Extract content from a chunk in the stream.

        :param chunk: A chunk from the stream
        :return: The content from the chunk, or None if no content
        """

    async def _run_streaming(
        self,
        messages: List[Message],
        queue: asyncio.Queue,
        stop_list: List[str],
        buffer_size: int = 20,  # Number of characters to buffer
        buffer_time: float = 1.0,  # Time in seconds to wait before flushing buffer
    ) -> None:
        """
        Send messages to WebSocket and ensure they are sent in real-time with buffering.

        :param messages: The list of messages to process.
        :param queue: The queue to store the response.
        :param stop_list: The list of stop words to use for the model.
        :param buffer_size: Number of characters to buffer before sending.
        :param buffer_time: Maximum time to wait before flushing buffer.
        """
        if not await self.connect():
            logger.error("Failed to connect to WebSocket, aborting...")
            await queue.put("Error: Could not connect to WebSocket.")
            return

        msg = [m.model_dump() for m in messages]
        if not msg:
            if self._sio is not None and self._sio.connected:
                await self._sio.emit(
                    "message", json.dumps({"error": "No messages to process."})
                )
            await queue.put("No messages to process.")
            return

        try:
            # Apply retry decorator to _process_stream
            @self._get_retry_decorator()
            async def _process_stream_with_retry(
                messages: List[Dict], stop_list: List[str]
            ) -> Any:
                return await self._process_stream(messages, stop_list)

            stream = await _process_stream_with_retry(msg, stop_list)

            full_response = ""
            buffer = ""
            last_flush_time = asyncio.get_event_loop().time()

            async def flush_buffer() -> None:
                nonlocal buffer, last_flush_time
                if buffer:
                    if self._sio is not None and self._sio.connected:
                        await self._sio.emit("message", json.dumps({"content": buffer}))
                        last_flush_time = asyncio.get_event_loop().time()
                        buffer = ""
                    else:
                        logger.error("Lost connection to server, stopping stream.")
                        raise ConnectionError("Lost connection to WebSocket server")

            try:
                for chunk in stream:
                    content = self._extract_content_from_chunk(chunk)

                    if content:
                        full_response += content
                        buffer += content

                        # Check if we need to flush the buffer
                        current_time = asyncio.get_event_loop().time()
                        time_since_last_flush = current_time - last_flush_time

                        # Flush if buffer is full or enough time has passed
                        if (
                            len(buffer) >= buffer_size
                            or time_since_last_flush >= buffer_time
                        ):
                            await flush_buffer()

                    # Small yield to allow other tasks to run
                    await asyncio.sleep(0)

                # Flush any remaining content in the buffer
                await flush_buffer()

                # Send full response
                if self._sio is not None and self._sio.connected:
                    await self._sio.emit(
                        "message", json.dumps({"full_response": full_response})
                    )
                else:
                    logger.error(
                        "Lost connection to server, could not send full_response."
                    )
                await asyncio.sleep(0)
                await queue.put(full_response)
            except Exception as e:
                logger.error(f"Error during streaming: {e}")
                await queue.put(f"Error during streaming: {str(e)}")
                raise
        finally:
            await self.disconnect()
            await asyncio.sleep(0)

    async def connect(self) -> bool:
        """
        Ensure the Socket.IO client is properly connected before sending messages.
        Uses tenacity for exponential backoff retry logic.
        """
        if not self._stream_url:
            logger.error("Stream URL is not set or empty.")
            return False

        # Initialize the Socket.IO client if not already created
        if self._sio is None:
            self._sio = socketio.AsyncClient(reconnection=True, reconnection_attempts=5)

        # If already connected, no need to reconnect
        if self._sio is not None and self._sio.connected:
            logger.info("Already connected to WebSocket server")
            return True

        # Define retry decorator for connection attempts
        @retry(
            stop=stop_after_attempt(5),  # Try 5 times
            wait=wait_exponential(multiplier=1, min=1, max=10),  # Exponential backoff
            retry=retry_if_exception_type(
                (
                    socketio.exceptions.ConnectionError,
                    asyncio.TimeoutError,
                    Exception,
                )
            ),
            reraise=False,  # Don't reraise the exception, we'll handle it
        )
        async def _connect_with_retry() -> bool:
            """
            Attempt to connect to the WebSocket server with exponential backoff.
            """
            sio_state = self._sio.eio.state if self._sio is not None else "N/A"
            logger.debug(f"Connecting to WebSocket server... Current state {sio_state}")

            if self._sio is not None:
                await self._sio.connect(self._stream_url, wait_timeout=5)
                logger.info("Connected to WebSocket server")
                return True
            return False

        try:
            result = await _connect_with_retry()
            return result
        except Exception as e:
            logger.error(
                f"Failed to connect to WebSocket server after multiple attempts: {e}"
            )
            return False

    async def disconnect(self) -> None:
        """
        Properly disconnect the Socket.IO client.
        """
        if self._sio and self._sio.connected:
            try:
                await self._sio.disconnect()
                logger.info("Disconnected from WebSocket server")
            except socketio.exceptions.ConnectionError as e:
                logger.error(f"Error during disconnect: {e}")

        # Ensure full cleanup before next connection attempt
        await asyncio.sleep(1.0)
        self._sio = None
