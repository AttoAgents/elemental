"""
Abstract class for the Language Model (LLM) in the Atomic framework.
"""

import asyncio
import json
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional

import requests.exceptions
import socketio
from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from elemental_agents.llm.data_model import Message, ModelParameters


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
        """
        if not self._stream_url:
            logger.error("Stream URL is not set or empty.")
            return False

        # Initialize the Socket.IO client if not already created
        if self._sio is None:
            self._sio = socketio.AsyncClient(reconnection=True, reconnection_attempts=5)

        sio_state = self._sio.eio.state if self._sio is not None else "N/A"
        logger.debug(
                ("Connecting to WebSocket server... " 
                f"Current state {sio_state}")
            )

        # If already connected, no need to reconnect
        if self._sio is not None and self._sio.connected:
            logger.info("Already connected to WebSocket server")
            return True

        # try:
        #     if self._sio is not None:
        #         await self._sio.connect(self._stream_url, wait_timeout=5)
        #     logger.info("Connected to WebSocket server")
        #     return True
        # except socketio.exceptions.ConnectionError as e:
        #     logger.error(f"Connection failed: {e}")
        #     return False

        max_retries = 5
        base_delay = 1  # seconds

        for attempt in range(1, max_retries + 1):
            try:
                logger.debug(f"Connection attempt {attempt} to {self._stream_url}")
                if self._sio is not None:
                    await self._sio.connect(self._stream_url, wait_timeout=5)
                    logger.info("Connected to WebSocket server")
                    return True
            except socketio.exceptions.ConnectionError as e:
                logger.warning(f"Connection attempt {attempt} failed: {e}")
            except asyncio.TimeoutError:
                logger.warning(f"Connection attempt {attempt} timed out.")
            except Exception as e:
                logger.error(
                    f"Unexpected error during connection attempt {attempt}: {e}"
                )

            # Exponential backoff delay before next retry
            delay = base_delay * (2 ** (attempt - 1))
            logger.info(f"Retrying in {delay} seconds...")
            await asyncio.sleep(delay)

        logger.error(
            f"Failed to connect to WebSocket server after {max_retries} attempts."
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
