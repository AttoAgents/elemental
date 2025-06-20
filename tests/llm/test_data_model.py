"""
Comprehensive test cases for data model classes.
"""

import base64
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from elemental_agents.llm.data_model import (
    ImageContent,
    Message,
    ModelParameters,
    TextContent,
)


class TestTextContent(unittest.TestCase):
    """
    Test TextContent class.
    """

    def test_text_content_creation(self) -> None:
        """
        Test creating TextContent with text.
        """
        content = TextContent(text="Hello world")

        self.assertEqual(content.type, "text")
        self.assertEqual(content.text, "Hello world")

    def test_text_content_default_type(self) -> None:
        """
        Test TextContent has default type 'text'.
        """
        content = TextContent(text="Test")

        self.assertEqual(content.type, "text")


class TestImageContent(unittest.TestCase):
    """
    Test ImageContent class.
    """

    def setUp(self) -> None:
        """
        Set up test fixtures.
        """
        # Create a temporary image file for testing
        self.temp_dir = tempfile.mkdtemp()
        self.test_image_path = os.path.join(self.temp_dir, "test_image.png")

        # Create a simple PNG-like binary data
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

    def test_image_content_creation_with_base64(self) -> None:
        """
        Test creating ImageContent with base64 data.
        """
        base64_data = base64.b64encode(self.test_image_data).decode("utf-8")
        content = ImageContent(base64_data=base64_data, media_type="image/png")

        self.assertEqual(content.type, "image")
        self.assertEqual(content.base64_data, base64_data)
        self.assertEqual(content.media_type, "image/png")
        self.assertIsNone(content.file_path)

    def test_image_content_creation_with_file_path(self) -> None:
        """
        Test creating ImageContent with file path.
        """
        content = ImageContent(file_path=self.test_image_path, media_type="image/png")

        self.assertEqual(content.type, "image")
        self.assertEqual(content.file_path, self.test_image_path)
        self.assertEqual(content.media_type, "image/png")
        self.assertIsNone(content.base64_data)

    def test_get_base64_from_base64_data(self) -> None:
        """
        Test getting base64 data when base64_data is set.
        """
        base64_data = base64.b64encode(self.test_image_data).decode("utf-8")
        content = ImageContent(base64_data=base64_data)

        result = content.get_base64()
        self.assertEqual(result, base64_data)

    def test_get_base64_from_file_path(self) -> None:
        """
        Test getting base64 data when file_path is set.
        """
        content = ImageContent(file_path=self.test_image_path)

        result = content.get_base64()
        expected = base64.b64encode(self.test_image_data).decode("utf-8")
        self.assertEqual(result, expected)

    def test_get_base64_no_data_raises_error(self) -> None:
        """
        Test that get_base64 raises error when no data available.
        """
        content = ImageContent()

        with self.assertRaises(ValueError) as context:
            content.get_base64()

        self.assertIn("No valid image data", str(context.exception))

    def test_get_base64_nonexistent_file_raises_error(self) -> None:
        """
        Test that get_base64 raises error for nonexistent file.
        """
        content = ImageContent(file_path="/nonexistent/path.png")

        with self.assertRaises(ValueError):
            content.get_base64()

    def test_get_file_path(self) -> None:
        """
        Test getting file path.
        """
        content = ImageContent(file_path=self.test_image_path)

        result = content.get_file_path()
        self.assertEqual(result, self.test_image_path)

    def test_get_file_path_none(self) -> None:
        """
        Test getting file path when None.
        """
        content = ImageContent(base64_data="test")

        result = content.get_file_path()
        self.assertIsNone(result)

    def test_default_media_type(self) -> None:
        """
        Test default media type is image/png.
        """
        content = ImageContent()

        self.assertEqual(content.media_type, "image/png")


class TestMessage(unittest.TestCase):
    """
    Test Message class.
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

    def test_message_creation_with_string_content(self) -> None:
        """
        Test creating Message with string content.
        """
        message = Message(role="user", content="Hello world")

        self.assertEqual(message.role, "user")
        self.assertEqual(message.content, "Hello world")

    def test_message_creation_with_structured_content(self) -> None:
        """
        Test creating Message with structured content.
        """
        content = [TextContent(text="Hello"), ImageContent(base64_data="test_data")]
        message = Message(role="user", content=content)

        self.assertEqual(message.role, "user")
        self.assertEqual(len(message.content), 2)
        self.assertIsInstance(message.content[0], TextContent)
        self.assertIsInstance(message.content[1], ImageContent)

    def test_add_image_to_string_content(self) -> None:
        """
        Test adding image to message with string content.
        """
        message = Message(role="user", content="Hello")
        message.add_image(self.test_image_path)

        self.assertIsInstance(message.content, list)
        self.assertEqual(len(message.content), 2)
        self.assertIsInstance(message.content[0], TextContent)
        self.assertIsInstance(message.content[1], ImageContent)
        self.assertEqual(message.content[0].text, "Hello")

    def test_add_image_from_file_path(self) -> None:
        """
        Test adding image from file path.
        """
        message = Message(role="user", content="Hello")
        message.add_image(self.test_image_path)

        images = message.get_images()
        self.assertEqual(len(images), 1)
        self.assertEqual(images[0].file_path, self.test_image_path)
        self.assertEqual(images[0].media_type, "image/png")

    def test_add_image_from_path_object(self) -> None:
        """
        Test adding image from Path object.
        """
        message = Message(role="user", content="Hello")
        message.add_image(Path(self.test_image_path))

        images = message.get_images()
        self.assertEqual(len(images), 1)
        self.assertEqual(images[0].file_path, self.test_image_path)

    def test_add_image_from_bytes(self) -> None:
        """
        Test adding image from bytes data.
        """
        message = Message(role="user", content="Hello")
        message.add_image(self.test_image_data)

        images = message.get_images()
        self.assertEqual(len(images), 1)
        expected_base64 = base64.b64encode(self.test_image_data).decode("utf-8")
        self.assertEqual(images[0].base64_data, expected_base64)

    @patch("requests.get")
    def test_add_image_from_url(self, mock_get: Mock) -> None:
        """
        Test adding image from URL.
        """
        mock_response = Mock()
        mock_response.content = self.test_image_data
        mock_response.headers = {"content-type": "image/png"}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        message = Message(role="user", content="Hello")
        message.add_image("https://example.com/image.png")

        images = message.get_images()
        self.assertEqual(len(images), 1)
        expected_base64 = base64.b64encode(self.test_image_data).decode("utf-8")
        self.assertEqual(images[0].base64_data, expected_base64)
        self.assertEqual(images[0].media_type, "image/png")

    @patch("requests.get")
    def test_add_image_from_url_with_extension_fallback(self, mock_get: Mock) -> None:
        """
        Test adding image from URL with media type from extension.
        """
        mock_response = Mock()
        mock_response.content = self.test_image_data
        mock_response.headers = {"content-type": "text/html"}  # Wrong content type
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        message = Message(role="user", content="Hello")
        message.add_image("https://example.com/image.jpg")

        images = message.get_images()
        self.assertEqual(len(images), 1)
        self.assertEqual(images[0].media_type, "image/jpeg")

    @patch("requests.get")
    def test_add_image_from_url_request_failure(self, mock_get: Mock) -> None:
        """
        Test adding image from URL when request fails.
        """
        mock_get.side_effect = Exception("Network error")

        message = Message(role="user", content="Hello")

        with self.assertRaises(ValueError) as context:
            message.add_image("https://example.com/image.png")

        self.assertIn("Failed to download image from URL", str(context.exception))

    def test_add_image_nonexistent_file(self) -> None:
        """
        Test adding image from nonexistent file path.
        """
        message = Message(role="user", content="Hello")

        with self.assertRaises(ValueError) as context:
            message.add_image("/nonexistent/path.png")

        self.assertIn("File path does not exist", str(context.exception))

    def test_add_image_invalid_source(self) -> None:
        """
        Test adding image with invalid source type.
        """
        message = Message(role="user", content="Hello")

        with self.assertRaises(ValueError) as context:
            message.add_image(123)  # Invalid type

        self.assertIn("Image source must be", str(context.exception))

    def test_is_multimodal_string_content(self) -> None:
        """
        Test is_multimodal with string content.
        """
        message = Message(role="user", content="Hello")

        self.assertFalse(message.is_multimodal())

    def test_is_multimodal_with_images(self) -> None:
        """
        Test is_multimodal with images.
        """
        message = Message(role="user", content="Hello")
        message.add_image(self.test_image_path)

        self.assertTrue(message.is_multimodal())

    def test_is_multimodal_text_only_structured(self) -> None:
        """
        Test is_multimodal with structured content but no images.
        """
        content = [TextContent(text="Hello")]
        message = Message(role="user", content=content)

        self.assertFalse(message.is_multimodal())

    def test_get_text_content_string(self) -> None:
        """
        Test getting text content from string message.
        """
        message = Message(role="user", content="Hello world")

        result = message.get_text_content()
        self.assertEqual(result, "Hello world")

    def test_get_text_content_structured(self) -> None:
        """
        Test getting text content from structured message.
        """
        content = [
            TextContent(text="Hello"),
            TextContent(text="world"),
            ImageContent(base64_data="test"),
        ]
        message = Message(role="user", content=content)

        result = message.get_text_content()
        self.assertEqual(result, "Hello world")

    def test_get_text_content_dict_format(self) -> None:
        """
        Test getting text content from dict format.
        """
        content = [
            {"type": "text", "text": "Hello"},
            {"type": "text", "text": "world"},
            {"type": "image", "data": "test"},
        ]
        message = Message(role="user", content=content)

        result = message.get_text_content()
        self.assertEqual(result, "Hello world")

    def test_get_images_string_content(self) -> None:
        """
        Test getting images from string content.
        """
        message = Message(role="user", content="Hello")

        images = message.get_images()
        self.assertEqual(len(images), 0)

    def test_get_images_with_images(self) -> None:
        """
        Test getting images from message with images.
        """
        message = Message(role="user", content="Hello")
        message.add_image(self.test_image_path)
        message.add_image(self.test_image_data)

        images = message.get_images()
        self.assertEqual(len(images), 2)
        self.assertIsInstance(images[0], ImageContent)
        self.assertIsInstance(images[1], ImageContent)


class TestModelParameters(unittest.TestCase):
    """
    Test ModelParameters class.
    """

    def test_model_parameters_defaults(self) -> None:
        """
        Test ModelParameters with default values.
        """
        params = ModelParameters()

        self.assertEqual(params.temperature, 0.0)
        self.assertIsNone(params.stop)
        self.assertEqual(params.max_tokens, 1000)
        self.assertEqual(params.frequency_penalty, 0.0)
        self.assertEqual(params.presence_penalty, 0.0)
        self.assertEqual(params.top_p, 1.0)
        self.assertIsNone(params.reasoning_effort)
        self.assertFalse(params.thinking_enabled)
        self.assertIsNone(params.thinking_budget_tokens)

    def test_model_parameters_custom_values(self) -> None:
        """
        Test ModelParameters with custom values.
        """
        params = ModelParameters(
            temperature=0.7,
            stop=["STOP", "END"],
            max_tokens=2000,
            frequency_penalty=0.5,
            presence_penalty=0.3,
            top_p=0.9,
            reasoning_effort="high",
            thinking_enabled=True,
            thinking_budget_tokens=1500,
        )

        self.assertEqual(params.temperature, 0.7)
        self.assertEqual(params.stop, ["STOP", "END"])
        self.assertEqual(params.max_tokens, 2000)
        self.assertEqual(params.frequency_penalty, 0.5)
        self.assertEqual(params.presence_penalty, 0.3)
        self.assertEqual(params.top_p, 0.9)
        self.assertEqual(params.reasoning_effort, "high")
        self.assertTrue(params.thinking_enabled)
        self.assertEqual(params.thinking_budget_tokens, 1500)


if __name__ == "__main__":
    unittest.main()
