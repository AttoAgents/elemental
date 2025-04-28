"""
Abbreviations resolver tool.
"""

import json
import os
from pathlib import Path

import requests
from loguru import logger
from pydantic import Field

from elemental_agents.core.toolbox.tool import (
    Tool,
    ToolInitParameters,
    ToolParameters,
    ToolResult,
)
from elemental_agents.utils.exceptions import ToolException


class AbbreviationsFetchError(ToolException):
    """
    Abbreviations fetch errors.
    """


class AbbreviationsResolverParams(ToolParameters):
    """
    Parameters for the AbbreviationsResolver tool

    :param abbreviation: The abreviation to resolve
    """

    abbreviation: str = Field(..., description="The abbreviation to resolve")


class AbbreviationsResolverResult(ToolResult):
    """
    Result for the AbbreviationsResolver tool

    :param result: The resolved abbreviation
    """

    result: str = Field(..., description="The resolved abbreviation")

    def __str__(self) -> str:
        return f"{self.result}"


class AbbreviationsResolverInitParameters(ToolInitParameters):
    """
    Initialization parameters for the AbbreviationsResolver tool. Two sources of
    abbreviations are supported: 1. The abbreviations.json file in the toolbox
    directory. 2. The abbreviations service url with the abbreviation as a query
    parameter.
    """

    source_file: str = Field(
        description="The source of abbreviations", default="abbreviations.json"
    )
    url: str | None = Field(
        description="The url of the abbreviations service", default=None
    )


class AbbreviationsResolver(Tool):
    """
    Tool for resolving abbreviations to their full form. Two sources of
    abbreviations are supported: 1. The abbreviations.json file in the toolbox
    directory. 2. The abbreviations service url with the abbreviation as a query
    parameter. Tool is initialized with one of these two sources.
    """

    name = "AbbreviationsResolver"
    description = (
        "Provides the full form of an abbreviation using a local file or a service."
    )

    def __init__(
        self,
        init_params: AbbreviationsResolverInitParameters = AbbreviationsResolverInitParameters(),
    ) -> None:
        self._abbreviations_file = init_params.source_file
        self._abbreviations_url = init_params.url

    def run(
        self, parameters: AbbreviationsResolverParams  # type: ignore
    ) -> AbbreviationsResolverResult:
        """
        Run the abbreviations resolver tool with the given parameters.

        :param parameters: The abbreviation to resolve
        :return: The resolved abbreviation
        """

        atomic_workspace = os.getenv("ATOMIC_WORKSPACE", "./")
        work_directory = Path(atomic_workspace).resolve()  # Resolve absolute path

        # Change the current working directory to ATOMIC_WORKSPACE
        os.chdir(work_directory)
        logger.info(f"Changed working directory to: {work_directory}")

        # Search in the abbreviations file
        data = {}
        if self._abbreviations_file:
            # Check if the path exists
            try:
                if os.path.exists(self._abbreviations_file):
                    with open(self._abbreviations_file, "r", encoding="utf-8") as file:
                        data = json.load(file)
                else:
                    logger.error("The abbreviations file does not exist")
                    raise ToolException("The abbreviations file does not exist")
            except json.JSONDecodeError as e:
                logger.error("The abbreviations file is not a valid JSON file")
                raise ToolException(
                    "The abbreviations file is not a valid JSON file"
                ) from e

            # Check if the abbreviation is in the file
            if parameters.abbreviation in data:
                result = data[parameters.abbreviation]
                final_result = AbbreviationsResolverResult(result=result)
                return final_result

        # query the abbreviation service
        if self._abbreviations_url:
            try:
                response = requests.get(
                    self._abbreviations_url,
                    params={"abbreviation": parameters.abbreviation},
                    timeout=5,
                )
                response.raise_for_status()
            except requests.exceptions.Timeout as e:
                raise ToolException("Request timed out after 5 seconds") from e
            except requests.exceptions.RequestException as e:
                raise AbbreviationsFetchError(
                    f"Failed to fetch the abbreviation {parameters.abbreviation}."
                ) from e

            data = response.json()
            result = data.get("result", "")
            final_result = AbbreviationsResolverResult(result=result)
            return final_result

        result = "The full form of the abbreviation is not available"
        final_result = AbbreviationsResolverResult(result=result)
        return final_result
