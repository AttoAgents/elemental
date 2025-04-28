"""
Tool to list the content of a directory listing everything or only the files.
"""

import os
from pathlib import Path
from typing import List

from loguru import logger
from pydantic import Field

from elemental_agents.core.toolbox.tool import Tool, ToolParameters, ToolResult


# Define specific Pydantic model for ListFiles input
class ListFilesParams(ToolParameters):
    """
    Defines the input parameters for the ListFiles tool.

    :param path: Directory path to list files
    :param files_only: Whether to list only files or include directories
    """

    path: str = Field(default=".", description="Directory path to list files")
    files_only: bool = Field(
        default=True, description="Whether to list only files or include directories"
    )


class ListFilesResult(ToolResult):
    """
    Defines the output result for the ListFiles tool.

    :param files: List of files in the directory
    """

    files: List[str] = Field(default=[], description="List of files in the directory")
    status: str = Field(default="OK", description="The status of the operation")

    def __str__(self) -> str:
        return f"Files: {self.files}, Status: {self.status}"

    def is_in_list(self, filename: str) -> bool:
        """
        Check if a specific file is in the list of files.

        :param filename: Name of the file to check
        :return: True if file is in the list, False otherwise
        """
        if isinstance(self.files, list):  # Ensure files is a list
            return filename in self.files
        return False


# ListFiles tool with class-level name and description
class ListFiles(Tool):
    """
    Tool to list the content of a directory listing everything or only the files.
    """

    name = "ListFiles"
    description = "Lists files in a directory."
    example = {"name": "ListFiles", "parameters": {"path": ".", "files_only": True}}

    def run(self, parameters: ListFilesParams) -> ListFilesResult:  # type: ignore
        """
        List files in the specified directory.

        :param parameters: ListFilesParams object with the path and files_only flag
        :return: List of files in the directory
        """

        atomic_workspace = os.getenv("ATOMIC_WORKSPACE", "./")
        work_directory = Path(atomic_workspace).resolve()  # Resolve absolute path

        # Change the current working directory to ATOMIC_WORKSPACE
        os.chdir(work_directory)
        logger.info(f"Changed working directory to: {work_directory}")

        fns = []

        try:
            # List files and directories in the specified path
            files = os.listdir(parameters.path)

            for file in files:
                file_path = os.path.join(parameters.path, file)
                if parameters.files_only and os.path.isfile(file_path):
                    fns.append(file)
                elif not parameters.files_only:
                    fns.append(file)

            final_result = ListFilesResult(files=fns, status="OK")
            return final_result

        except (OSError, TypeError) as e:
            logger.error(f"Error listing files: {e}")
            final_result = ListFilesResult(files=[], status=f"Error {e}")
            return final_result


if __name__ == "__main__":

    # Test the ListFiles tool
    tool = ListFiles()
    params = ListFilesParams(path=".", files_only=True)
    result = tool.run(params)
    logger.info(result)
