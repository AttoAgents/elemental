"""
Tool class definition. Tools are the building blocks of the toolbox and are used
to perform specific tasks.
"""

import json
from abc import ABC, abstractmethod
from typing import Any, Dict

from pydantic import BaseModel


class ToolParameters(BaseModel):
    """
    Base class for tool parameters. Tool parameters are used to specify the
    input parameters required by the tool.
    """


class ToolInitParameters(BaseModel):
    """
    Base class for tool initialization parameters. Tool initialization parameters
    are used to specify the input parameters required to initialize the tool.
    """


class ToolResult(BaseModel):
    """
    Base class for tool results. Tool results are used to specify the output
    parameters returned by the tool.
    """


class Tool(ABC):
    """
    Abstract base class for tools. Tools are the building blocks of the toolbox
    and are used to perform specific tasks.
    """

    name: str = None
    description: str = None
    example: Dict[str, Any] = None

    def __init__(self, init_params: BaseModel = None) -> None:
        """
        Constructor for the tool. Initializes the tool with the specified
        parameters. Optional as not all tools require initialization.

        :param init_params: Pydantic model with the tool's initialization
            parameters
        """

    @abstractmethod
    def run(self, parameters: ToolParameters) -> ToolResult:
        """
        Abstract method to run the tool with the specified parameters.
        Parameters are provided as a Pydantic model.

        :param parameters: Pydantic model with the tool's parameters
        :return: Result of the tool execution
        """

    @classmethod
    def get_name(cls) -> str:
        """
        Return the name of the tool.

        :return: Name of the tool
        """

        return cls.name

    @classmethod
    def get_description(cls) -> str:
        """
        Return the description of the tool.

        :return: Description of the tool
        """

        return cls.description

    @classmethod
    def get_example(cls) -> str:
        """
        Return the example of the tool.

        :return: Example of the tool
        """
        if cls.example is None:
            return None
        return json.dumps(cls.example)
