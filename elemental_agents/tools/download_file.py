"""
Tool download file from URL and save it to a specified path.
"""

import os
from pathlib import Path

import requests
from loguru import logger
from pydantic import Field

from elemental_agents.core.toolbox.tool import Tool, ToolParameters, ToolResult
from elemental_agents.utils.utils import is_path_in_current_or_subdirectory


class DownloadFileParams(ToolParameters):
    """
    Defines the input parameters for the DownloadFile tool.

    :param url: URL of the file to download
    :param save_path: Path where the file should be saved
    """

    url: str = Field(..., description="URL of the file to download.")
    save_path: str = Field(
        default="./downloaded_file", description="Path where the file should be saved."
    )


class DownloadFileResult(ToolResult):
    """
    Defines the output result for the DownloadFile tool.
    """

    filename: str = Field(
        default="", description="Name of the file that was downloaded."
    )
    status: str = Field(
        default="OK", description="Indicates if the file was downloaded successfully."
    )
    message: str = Field(
        default="", description="Message regarding the download process."
    )

    def __str__(self) -> str:

        return (
            f"{{ 'FileWritten': '{self.filename}', "
            f"'status': '{self.status}', "
            f"'message': '{self.message}' }}"
        )


class DownloadFile(Tool):
    """
    Tool to download a file from a given URL and save it to a specified path.
    """

    name = "DownloadFile"
    description = "Downloads a file from a URL and saves it to a specified path."

    def run(self, parameters: DownloadFileParams) -> DownloadFileResult:  # type: ignore
        """
        Download the file from the specified URL and save it to the given path.

        :param parameters: DownloadFileParams object with the URL and save path
        :return: DownloadFileResult indicating success or failure
        """

        atomic_workspace = os.getenv("ATOMIC_WORKSPACE", "./")
        work_directory = Path(atomic_workspace).resolve()  # Resolve absolute path

        # Change the current working directory to ATOMIC_WORKSPACE
        os.chdir(work_directory)
        logger.info(f"Changed working directory to: {work_directory}")

        try:
            # Make a request to the URL
            response = requests.get(parameters.url, stream=True, timeout=10)
            response.raise_for_status()  # Raise an error for bad responses

            # Check if the path is in the subdirectories of the current directory
            save_path = Path(parameters.save_path)
            if not is_path_in_current_or_subdirectory(str(save_path)):
                logger.error(f"Path {save_path} is not in the current or subdirectory")
                raise ValueError(
                    f"Path {save_path} is not in the current or subdirectory"
                )

            # Write the content to the file
            with open(save_path, "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)

            logger.info(f"File downloaded successfully and saved to: {save_path}")

            result = DownloadFileResult(
                filename=str(save_path), status="OK", message="File saved successfully."
            )
            return result

        except requests.exceptions.RequestException as e:
            logger.error(f"Error downloading file: {e}")
            result = DownloadFileResult(status="ERROR", message=str(e))
            return result
