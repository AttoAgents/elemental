"""
Data model for ModelParameters and Message type.
"""

import base64
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from loguru import logger
from pydantic import BaseModel, Field


class TextContent(BaseModel):
    """
    Represents a text content part in a message.
    """

    type: str = "text"
    text: str


class ImageContent(BaseModel):
    """
    Represents an image content part in a message.
    Stores images as either base64 data or file path for local processing.
    """

    type: str = "image"
    # Internal storage - always one of these
    base64_data: Optional[str] = None
    file_path: Optional[str] = None
    media_type: str = "image/png"

    def get_base64(self) -> str:
        """
        Get base64 representation of the image.

        :return: Base64 encoded image data
        """
        if self.base64_data:
            return self.base64_data
        elif self.file_path and os.path.exists(self.file_path):
            with open(self.file_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
        else:
            raise ValueError("No valid image data or file path available")

    def get_file_path(self) -> Optional[str]:
        """
        Get file path if available.

        :return: File path or None
        """
        return self.file_path


# Define a type for message content parts
ContentPart = Union[TextContent, ImageContent]


class Message(BaseModel):
    """
    Message class that represents a message in framework.
    Supports both simple text messages and complex messages with text and images.
    """

    role: str
    content: Union[str, List[ContentPart]] = Field(
        ...,
        description="Content can be a simple string or a list of content parts (text and images)",
    )

    def add_image(self, image_source: Union[str, bytes, Path]) -> None:
        """
        Add an image to the message content.

        :param image_source: Can be:
            - URL string (will be downloaded and converted to base64)
            - File path string or Path object
            - Raw bytes data
        """
        # Convert string content to structured content if needed
        if isinstance(self.content, str):
            text_content = self.content
            self.content = [TextContent(text=text_content)]

        image_content = None

        if isinstance(image_source, (str, Path)):
            image_source_str = str(image_source)

            if image_source_str.startswith(("http://", "https://")):
                # It's a URL - download and convert to base64
                try:
                    import requests

                    response = requests.get(image_source_str, timeout=30)
                    response.raise_for_status()

                    # Determine media type from response headers or URL
                    content_type = response.headers.get("content-type", "image/png")
                    if not content_type.startswith("image/"):
                        # Try to guess from URL extension
                        if image_source_str.lower().endswith((".jpg", ".jpeg")):
                            content_type = "image/jpeg"
                        elif image_source_str.lower().endswith(".png"):
                            content_type = "image/png"
                        elif image_source_str.lower().endswith(".gif"):
                            content_type = "image/gif"
                        elif image_source_str.lower().endswith(".webp"):
                            content_type = "image/webp"
                        else:
                            content_type = "image/png"  # default

                    base64_data = base64.b64encode(response.content).decode("utf-8")
                    image_content = ImageContent(
                        base64_data=base64_data, media_type=content_type
                    )
                    logger.debug(
                        f"Downloaded and converted URL to base64: {image_source_str}"
                    )

                except Exception as e:
                    logger.error(
                        f"Failed to download image from URL {image_source_str}: {e}"
                    )
                    raise ValueError(f"Failed to download image from URL: {e}")

            elif os.path.exists(image_source_str):
                # It's a local file path
                # Determine media type from file extension
                ext = Path(image_source_str).suffix.lower()
                media_type_map = {
                    ".jpg": "image/jpeg",
                    ".jpeg": "image/jpeg",
                    ".png": "image/png",
                    ".gif": "image/gif",
                    ".webp": "image/webp",
                    ".bmp": "image/bmp",
                }
                media_type = media_type_map.get(ext, "image/png")

                image_content = ImageContent(
                    file_path=image_source_str, media_type=media_type
                )
                logger.debug(f"Added local file path: {image_source_str}")
            else:
                raise ValueError(f"File path does not exist: {image_source_str}")

        elif isinstance(image_source, bytes):
            # Raw bytes data
            base64_data = base64.b64encode(image_source).decode("utf-8")
            image_content = ImageContent(
                base64_data=base64_data,
                media_type="image/png",  # default, could be improved with image type detection
            )
            logger.debug("Converted bytes data to base64")
        else:
            raise ValueError("Image source must be a URL, file path, or bytes data")

        # Add image to content
        if isinstance(self.content, list) and image_content:
            self.content.append(image_content)

    def is_multimodal(self) -> bool:
        """
        Check if the message contains multiple modalities (text and images).

        :return: True if the message contains images, False otherwise
        """
        if isinstance(self.content, list):
            return any(
                isinstance(part, ImageContent)
                or (isinstance(part, dict) and part.get("type") == "image")
                for part in self.content
            )
        return False

    def get_text_content(self) -> str:
        """
        Extract only the text content from the message.

        :return: Concatenated text content
        """
        if isinstance(self.content, str):
            return self.content

        text_parts = []
        for part in self.content:
            if isinstance(part, TextContent):
                text_parts.append(part.text)
            elif isinstance(part, dict) and part.get("type") == "text":
                text_parts.append(part.get("text", ""))

        return " ".join(text_parts)

    def get_images(self) -> List[ImageContent]:
        """
        Get all image content from the message.

        :return: List of ImageContent objects
        """
        if isinstance(self.content, str):
            return []

        images = []
        for part in self.content:
            if isinstance(part, ImageContent):
                images.append(part)
        return images


class ModelParameters(BaseModel):
    """
    ModelParameters class that represents the parameters for a language model.
    """

    temperature: float = 0.0
    stop: Optional[List[str]] = None
    max_tokens: int = 1000
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    top_p: float = 1.0
    # Reasoning parameters
    reasoning_effort: Optional[str] = None  # For OpenAI: "low", "medium", "high"
    thinking_enabled: bool = False  # For Anthropic and Ollama
    thinking_budget_tokens: Optional[int] = None  # For Anthropic
