"""
No action tool with the Elemental Knowledge Base API.
"""

from loguru import logger
from pydantic import Field

from elemental_agents.core.toolbox.tool import Tool, ToolParameters, ToolResult


class NoActionParams(ToolParameters):
    """
    Defines the input parameters for the NoAction tool.

    :param query: Query to search in the Knowledge Base
    """

    reason: str = Field(description="Reason for no action")


class NoActionResult(ToolResult):
    """
    Defines the output result for the NoAction tool.
    """

    results: str = Field(default="NoAction taken", description="NoAction taken comment")

    def __str__(self) -> str:
        return f"{self.results}"


class NoAction(Tool):
    """
    Tool to interact with the Elemental Knowledge Base API.
    """

    name = "NoAction"
    description = (
        "Tool selection for taking no action. "
        "To be used if no tool is applicable or no action is required."
    )
    example = {
        "name": "NoAction",
        "parameters": {
            "reason": "There is no need to execute any action at this moment.",
        },
    }

    def run(self, parameters: NoActionParams) -> NoActionResult:  # type: ignore
        """
        Search the Knowledge Base for the specified query.

        :param parameters: KnowledgeBaseParams object with the query
        :return: List of top results from the knowledge base
        """

        logger.info("NoAction taken")
        return NoActionResult()
