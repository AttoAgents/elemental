"""
Tool to read the content of a plain text file.
"""

import os
from pathlib import Path
from typing import Dict, List

from loguru import logger
from pydantic import Field

from elemental_agents.core.toolbox.tool import Tool, ToolParameters, ToolResult
from elemental_agents.utils.utils import is_path_in_current_or_subdirectory


# Define specific Pydantic model for ReadFiles input
class ReadFilesParams(ToolParameters):
    """
    Defines the input parameters for the ReadFiles tool.

    :param paths: List of file paths to read
    """

    paths: List[str] = Field(
        default=["./README"],
        description=(
            "List of file paths to read. All files need to be "
            "plain text and present in the working directory."
        ),
    )


class ReadFilesResult(ToolResult):
    """
    Defines the output result for the ReadFiles tool.
    """

    contents: Dict[str, str] = Field(
        default={}, description="Dictionary of file paths and their contents"
    )

    def __str__(self) -> str:
        return f"{self.contents}"


# ReadFiles tool with class-level name and description
class ReadFiles(Tool):
    """
    Tool to read the content of a file.
    """

    name = "ReadFiles"
    description = (
        "Reads plain text files from the current directory and provides their content."
    )

    def run(self, parameters: ReadFilesParams) -> ReadFilesResult:  # type: ignore
        """
        Read the content of the specified file. The files can be read only from
        subdirectories of the current directory. Only files are read, not
        directories. Files bigger than 1MB are not read.

        :param parameters: ReadFilesParams object with the path
        :return: Dictionary of the strings with the content of the files read
            from specified paths
        """

        atomic_workspace = os.getenv("ATOMIC_WORKSPACE", "./")
        work_directory = Path(atomic_workspace).resolve()  # Resolve absolute path

        # Change the current working directory to ATOMIC_WORKSPACE
        os.chdir(work_directory)
        logger.info(f"Changed working directory to: {work_directory}")

        contents = {}

        try:
            for path in parameters.paths:

                # Check if the path is in the subdirectories of the current directory
                if not is_path_in_current_or_subdirectory(path):
                    logger.error(f"Path {path} is not in the current or subdirectory")
                    continue

                # Check if the path is a file
                if os.path.isfile(path):

                    # Big files protection
                    file_size = os.path.getsize(path)
                    if file_size > 1000000:
                        logger.error(
                            f"File {path} is too big to read, size {file_size}"
                        )
                        continue

                    with open(path, "r", encoding="utf-8") as f:
                        contents[path] = f.read()
                else:
                    logger.error(f"Path {path} is not a file")

        except OSError as e:
            logger.error(f"Error reading files: {e}")

        result = ReadFilesResult(contents=contents)
        return result


if __name__ == "__main__":

    from rich.console import Console

    # Test the ReadFiles tool
    read_files = ReadFiles()
    params = ReadFilesParams(paths=["./README.md"])
    file_contents = read_files.run(params)

    console = Console()
    console.print(file_contents)
