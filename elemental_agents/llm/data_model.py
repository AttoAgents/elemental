"""
Data model for ModelParameters and Message type.
"""

from typing import List

from pydantic import BaseModel


class ModelParameters(BaseModel):
    """
    ModelParameters class that represents the parameters for a language model.
    """

    temperature: float = 0.0
    stop: List[str] = None
    max_tokens: int = 1000
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    top_p: float = 1.0


class Message(BaseModel):
    """
    Message class that represents a message in framework.
    """

    role: str
    content: str
