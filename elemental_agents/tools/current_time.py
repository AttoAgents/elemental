"""
Tool for getting the current date and time.
"""

from datetime import datetime

from loguru import logger
from pydantic import Field

from elemental_agents.core.toolbox.tool import Tool, ToolParameters, ToolResult


# Define specific Pydantic model for CurrentTime input
class CurrentTimeParams(ToolParameters):
    """
    Parameters for the CurrentTime tool

    :param format: The format string for the date and time
    """

    format: str = Field(
        "%Y-%m-%d %H:%M:%S",
        description="The format string for the date and time",
    )


class CurrentTimeResult(ToolResult):
    """
    Result for the CurrentTime tool

    :param current_time: The current date and time as a string
    """

    current_time: str = Field(..., description="The current date and time as a string")

    def __str__(self) -> str:
        return f"{self.current_time}"


# CurrentTime tool with class-level name and description
class CurrentTime(Tool):
    """
    Tool for getting the current date and time.
    """

    name = "CurrentTime"
    description = "Get the current date and time"

    def run(self, parameters: CurrentTimeParams) -> CurrentTimeResult:  # type: ignore
        """
        Run the current time tool with the given parameters.

        :param parameters: The parameters for the current time tool.
        :return: The current date and time as a string.
        """
        current_time = datetime.now().strftime(parameters.format)
        logger.info(f"Current time: {current_time}")

        return CurrentTimeResult(current_time=current_time)
