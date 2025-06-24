"""
Test cases for multimodal LLM functionality.
"""

import base64
import os
import tempfile
import unittest
from unittest.mock import Mock, patch

from elemental_agents.llm.data_model import ImageContent, Message, TextContent
from elemental_agents.llm.llm_mock import MockLLM
from elemental_agents.llm.llm_openai import OpenAILLM


class TestLLMMultimodal(unittest.TestCase):
    """
    Test multimodal functionality across LLM implementations.
    """

    def setUp(self) -> None:
        """
        Set up test fixtures.
        """
        self.temp_dir = tempfile.mkdtemp()
        self.test_image_path = os.path.join(self.temp_dir, "test_image.png")
        self.test_image_data = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde"

        with open(self.test_image_path, "wb") as f:
            f.write(self.test_image_data)

    def tearDown(self) -> None:
        """
        Clean up test fixtures.
        """
        if os.path.exists(self.test_image_path):
            os.remove(self.test_image_path)
        os.rmdir(self.temp_dir)

    def test_multimodal_message_creation(self) -> None:
        """
        Test creating multimodal messages.
        """
        message = Message(role="user", content="Describe this image")
        message.add_image(self.test_image_path)

        self.assertTrue(message.is_multimodal())
        self.assertEqual(len(message.get_images()), 1)
        self.assertEqual(message.get_text_content(), "Describe this image")

    def test_multimodal_message_with_multiple_images(self) -> None:
        """
        Test message with multiple images.
        """
        message = Message(role="user", content="Compare these images")
        message.add_image(self.test_image_path)
        message.add_image(self.test_image_data)

        self.assertTrue(message.is_multimodal())
        self.assertEqual(len(message.get_images()), 2)

        images = message.get_images()
        self.assertIsInstance(images[0], ImageContent)
        self.assertIsInstance(images[1], ImageContent)

    def test_multimodal_message_mixed_content(self) -> None:
        """
        Test message with mixed text and image content.
        """
        content = [
            TextContent(text="Look at this image:"),
            ImageContent(file_path=self.test_image_path),
            TextContent(text="What do you see?"),
        ]
        message = Message(role="user", content=content)

        self.assertTrue(message.is_multimodal())
        self.assertEqual(
            message.get_text_content(), "Look at this image: What do you see?"
        )
        self.assertEqual(len(message.get_images()), 1)

    def test_mock_llm_with_multimodal_message(self) -> None:
        """
        Test MockLLM handling multimodal messages.
        """
        llm = MockLLM()
        message = Message(role="user", content="Describe this image")
        message.add_image(self.test_image_path)

        result = llm.run([message])

        # MockLLM should handle multimodal messages gracefully
        self.assertIsNotNone(result)
        self.assertIn("user message", result)

    @patch("elemental_agents.llm.llm_openai.OpenAI")
    def test_openai_llm_multimodal_conversion(self, mock_openai: Mock) -> None:
        """
        Test OpenAI LLM multimodal message conversion.
        """
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "I see an image"
        mock_response.usage = None
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        llm = OpenAILLM(model_name="gpt-4o")

        message = {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this"},
                {
                    "type": "image",
                    "base64_data": base64.b64encode(self.test_image_data).decode(),
                    "media_type": "image/png",
                },
            ],
        }

        converted = llm._convert_message_for_openai(message)

        self.assertEqual(converted["role"], "user")
        self.assertEqual(len(converted["content"]), 2)
        self.assertEqual(converted["content"][0]["type"], "text")
        self.assertEqual(converted["content"][1]["type"], "image_url")
        self.assertIn(
            "data:image/png;base64,", converted["content"][1]["image_url"]["url"]
        )

    @patch("elemental_agents.llm.llm_openai.OpenAI")
    def test_openai_llm_multimodal_file_path_conversion(
        self, mock_openai: Mock
    ) -> None:
        """
        Test OpenAI LLM converting file path to base64.
        """
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "I see an image"
        mock_response.usage = None
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        llm = OpenAILLM(model_name="gpt-4o")

        message = {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this"},
                {
                    "type": "image",
                    "file_path": self.test_image_path,
                    "media_type": "image/png",
                },
            ],
        }

        converted = llm._convert_message_for_openai(message)

        self.assertEqual(converted["role"], "user")
        self.assertEqual(len(converted["content"]), 2)
        self.assertEqual(converted["content"][1]["type"], "image_url")

        # Should have converted file to base64
        expected_base64 = base64.b64encode(self.test_image_data).decode()
        self.assertIn(expected_base64, converted["content"][1]["image_url"]["url"])

    def test_multimodal_message_serialization(self) -> None:
        """
        Test that multimodal messages serialize correctly.
        """
        message = Message(role="user", content="Describe this image")
        message.add_image(self.test_image_path)

        # Test model_dump (Pydantic serialization)
        serialized = message.model_dump()

        self.assertEqual(serialized["role"], "user")
        self.assertIsInstance(serialized["content"], list)
        self.assertEqual(len(serialized["content"]), 2)

        # Should have text and image content
        content_types = [item["type"] for item in serialized["content"]]
        self.assertIn("text", content_types)
        self.assertIn("image", content_types)

    def test_image_content_media_type_detection(self) -> None:
        """
        Test media type detection for different image formats.
        """
        # Test different file extensions
        test_cases = [
            ("test.jpg", "image/jpeg"),
            ("test.jpeg", "image/jpeg"),
            ("test.png", "image/png"),
            ("test.gif", "image/gif"),
            ("test.webp", "image/webp"),
            ("test.bmp", "image/bmp"),
            ("test.unknown", "image/png"),  # default
        ]

        for filename, expected_media_type in test_cases:
            test_path = os.path.join(self.temp_dir, filename)
            with open(test_path, "wb") as f:
                f.write(self.test_image_data)

            try:
                message = Message(role="user", content="Test")
                message.add_image(test_path)

                images = message.get_images()
                self.assertEqual(len(images), 1)
                self.assertEqual(images[0].media_type, expected_media_type)
            finally:
                if os.path.exists(test_path):
                    os.remove(test_path)

    @patch("requests.get")
    def test_multimodal_message_from_url(self, mock_get: Mock) -> None:
        """
        Test adding image from URL.
        """
        mock_response = Mock()
        mock_response.content = self.test_image_data
        mock_response.headers = {"content-type": "image/png"}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        message = Message(role="user", content="Describe this image")
        message.add_image("https://example.com/image.png")

        self.assertTrue(message.is_multimodal())
        images = message.get_images()
        self.assertEqual(len(images), 1)
        self.assertEqual(images[0].media_type, "image/png")

        # Should have base64 data from URL
        expected_base64 = base64.b64encode(self.test_image_data).decode()
        self.assertEqual(images[0].base64_data, expected_base64)

    def test_multimodal_detection_edge_cases(self) -> None:
        """
        Test multimodal detection edge cases.
        """
        # String content - not multimodal
        message1 = Message(role="user", content="Hello")
        self.assertFalse(message1.is_multimodal())

        # List with only text - not multimodal
        message2 = Message(role="user", content=[TextContent(text="Hello")])
        self.assertFalse(message2.is_multimodal())

        # List with dict format text - not multimodal
        message3 = Message(role="user", content=[{"type": "text", "text": "Hello"}])
        self.assertFalse(message3.is_multimodal())

        # List with dict format image - multimodal
        message4 = Message(
            role="user",
            content=[
                {"type": "text", "text": "Hello"},
                {"type": "image", "data": "test"},
            ],
        )
        self.assertTrue(message4.is_multimodal())

    def test_get_text_content_edge_cases(self) -> None:
        """
        Test get_text_content with various content formats.
        """
        # Empty text parts
        message1 = Message(
            role="user",
            content=[
                TextContent(text=""),
                TextContent(text="Hello"),
                TextContent(text=""),
            ],
        )
        self.assertEqual(message1.get_text_content(), " Hello ")

        # Mixed with dict format
        message2 = Message(
            role="user",
            content=[
                {"type": "text", "text": "Hello"},
                {"type": "image", "data": "test"},
                {"type": "text", "text": "world"},
            ],
        )
        self.assertEqual(message2.get_text_content(), "Hello world")

        # No text content
        message3 = Message(role="user", content=[ImageContent(base64_data="test")])
        self.assertEqual(message3.get_text_content(), "")


if __name__ == "__main__":
    unittest.main()
