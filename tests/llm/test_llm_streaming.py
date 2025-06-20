"""
Test cases for LLM streaming functionality.
"""

import asyncio
import unittest
from unittest.mock import AsyncMock, Mock, patch
from typing import AsyncGenerator, Awaitable

from elemental_agents.llm.data_model import Message
from elemental_agents.llm.llm_mock import MockLLM


class TestLLMStreaming(unittest.TestCase):
    """
    Test LLM streaming functionality.
    """

    def setUp(self) -> None:
        """
        Set up test fixtures.
        """
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def tearDown(self) -> None:
        """
        Clean up test fixtures.
        """
        self.loop.close()

    async def test_streaming_basic_flow(self) -> None:
        """
        Test basic streaming flow.
        """
        llm = MockLLM(message_stream=True, stream_url="ws://localhost:8000")

        # Mock the WebSocket connection
        with patch.object(llm, "connect", return_value=True):
            with patch.object(llm, "_sio") as mock_sio:
                mock_sio.connected = True
                mock_sio.emit = AsyncMock()

                messages = [Message(role="user", content="Hello")]
                queue = asyncio.Queue()

                await llm._run_streaming(messages, queue, [])

                # Check that messages were emitted
                self.assertTrue(mock_sio.emit.called)

    async def test_streaming_connection_failure(self) -> None:
        """
        Test streaming when connection fails.
        """
        llm = MockLLM(message_stream=True, stream_url="ws://localhost:8000")

        # Mock failed connection
        with patch.object(llm, "connect", return_value=False):
            messages = [Message(role="user", content="Hello")]
            queue = asyncio.Queue()

            await llm._run_streaming(messages, queue, [])

            # Check that error message was queued
            result = await queue.get()
            self.assertIn("Error", result)

    async def test_streaming_buffer_management(self) -> None:
        """
        Test streaming buffer management.
        """
        llm = MockLLM(message_stream=True, stream_url="ws://localhost:8000")

        with patch.object(llm, "connect", return_value=True):
            with patch.object(llm, "_sio") as mock_sio:
                mock_sio.connected = True
                mock_sio.emit = AsyncMock()

                # Mock stream that produces content
                async def mock_stream() -> AsyncGenerator[Mock, None]:
                    """
                    Mock generator to simulate streaming events.
                    """
                    # Simulate a long message by yielding characters one by one
                    for char in "Hello world this is a long message":
                        yield Mock(content=char)

                with patch.object(llm, "_process_stream", return_value=mock_stream()):
                    with patch.object(
                        llm,
                        "_extract_content_from_chunk",
                        side_effect=lambda x: x.content,
                    ):
                        messages = [Message(role="user", content="Hello")]
                        queue = asyncio.Queue()

                        await llm._run_streaming(
                            messages, queue, [], buffer_size=5, buffer_time=0.1
                        )

                        # Check that emit was called multiple times (buffering)
                        self.assertGreater(mock_sio.emit.call_count, 1)

    def test_streaming_run_integration(self) -> None:
        """
        Test streaming run method integration.
        """
        llm = MockLLM(message_stream=True, stream_url="ws://localhost:8000")

        with patch.object(llm, "connect", return_value=True):
            with patch.object(llm, "_sio") as mock_sio:
                mock_sio.connected = True
                mock_sio.emit = AsyncMock()

                messages = [Message(role="user", content="Hello")]

                # This should trigger streaming mode
                result = llm.run(messages)

                self.assertIsNotNone(result)

    async def test_streaming_empty_messages(self) -> None:
        """
        Test streaming with empty messages.
        """
        llm = MockLLM(message_stream=True, stream_url="ws://localhost:8000")

        with patch.object(llm, "connect", return_value=True):
            with patch.object(llm, "_sio") as mock_sio:
                mock_sio.connected = True
                mock_sio.emit = AsyncMock()

                queue = asyncio.Queue()

                await llm._run_streaming([], queue, [])

                result = await queue.get()
                self.assertEqual(result, "No messages to process.")

    async def test_streaming_connection_lost_during_stream(self) -> None:
        """
        Test handling connection loss during streaming.
        """
        llm = MockLLM(message_stream=True, stream_url="ws://localhost:8000")

        with patch.object(llm, "connect", return_value=True):
            with patch.object(llm, "_sio") as mock_sio:
                # Start connected, then lose connection
                mock_sio.connected = True

                async def mock_emit(*args: str, **kwargs: str)-> None:
                    """
                    Mock emit function to simulate connection loss.
                    """
                    # Simulate connection loss
                    mock_sio.connected = False
                    raise ConnectionError("Connection lost")

                mock_sio.emit = mock_emit

                async def mock_stream() -> AsyncGenerator[Mock, None]:
                    """
                    Mock generator to simulate streaming events.
                    """
                    yield Mock(content="test")

                with patch.object(llm, "_process_stream", return_value=mock_stream()):
                    with patch.object(
                        llm, "_extract_content_from_chunk", return_value="test"
                    ):
                        messages = [Message(role="user", content="Hello")]
                        queue = asyncio.Queue()

                        with self.assertRaises(ConnectionError):
                            await llm._run_streaming(messages, queue, [])

    def run_async_test(self, coro: AsyncGenerator[None, None]) -> Awaitable[None]:
        """
        Helper to run async tests.
        """
        return self.loop.run_until_complete(coro)

    def test_streaming_basic_flow_sync(self) -> None:
        """
        Test basic streaming flow (sync wrapper).
        """
        self.run_async_test(self.test_streaming_basic_flow())

    def test_streaming_connection_failure_sync(self) -> None:
        """
        Test streaming connection failure (sync wrapper).
        """
        self.run_async_test(self.test_streaming_connection_failure())

    def test_streaming_buffer_management_sync(self) -> None:
        """
        Test streaming buffer management (sync wrapper).
        """
        self.run_async_test(self.test_streaming_buffer_management())

    def test_streaming_empty_messages_sync(self) -> None:
        """
        Test streaming with empty messages (sync wrapper).
        """
        self.run_async_test(self.test_streaming_empty_messages())

    def test_streaming_connection_lost_sync(self) -> None:
        """
        Test connection lost during streaming (sync wrapper).
        """
        self.run_async_test(self.test_streaming_connection_lost_during_stream())


if __name__ == "__main__":
    unittest.main()
