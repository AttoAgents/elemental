"""
Tool to write the content to a file.
"""

import os
from pathlib import Path
from typing import List

from loguru import logger
from pydantic import Field

from elemental_agents.core.toolbox.tool import Tool, ToolParameters, ToolResult
from elemental_agents.utils.utils import is_path_in_current_or_subdirectory


# Define specific Pydantic model for WriteFiles input
class WriteFileParams(ToolParameters):
    """
    Defines the input parameters for the WriteFile tool.

    :param paths: List of file paths to write
    :param contents: List of strings to write to the files
    """

    file_path: str = Field(
        default="./default-output.txt",
        description="Path of the file write to. Include the relative path and file name.",
    )
    content: List[str] = Field(
        default=["Default content line 1.", "Default content line 2."],
        description=(
            "List of strings to write to the file. "
            "Each line of the file is an element of this list. "
            "The list needs to be a valid Python list of strings. "
            "Never put comments that are not meant to be written to the file in this list."
        ),
    )


class WriteFileResult(ToolResult):
    """
    Defines the output result for the WriteFile tool.
    """

    path: str = Field(
        default="./default-output.txt",
        description="File paths that was requested to be written.",
    )
    status: str = Field(
        default="OK",
        description=(
            "Status of the file write operation. "
            "OK if the file was written successfully, ERROR if the "
            "file was not written successfully."
        ),
    )

    def __str__(self) -> str:
        return f"{{ 'FileWritten': '{self.path}', 'status': '{self.status}' }}"


class WriteFile(Tool):
    """
    Tool to write the content to a file.
    """

    name = "WriteFile"
    description = (
        "Writes file to local disk. Specify the file path "
        "and content to write. The file will be overwritten if it already exists."
    )
    example = {
        "name": "WriteFile",
        "parameters": {
            "file_path": "./example.py",
            "content": ["line 1", "line 2", "line 3"],
        },
    }

    def run(self, parameters: WriteFileParams) -> WriteFileResult:  # type: ignore
        """
        Write the content to the specified file. The files can be written only
        to subdirectories of the current directory. Only files are written,
        directories are not created. The files are overwritten if they already
        exist.

        :param parameters: WriteFilesParams object with the path and content
        :return: Dictionary with the path and status of the file write operation
        """

        atomic_workspace = os.getenv("ATOMIC_WORKSPACE", "./")
        work_directory = Path(atomic_workspace).resolve()  # Resolve absolute path

        # Change the current working directory to ATOMIC_WORKSPACE
        os.chdir(work_directory)
        logger.info(f"Changed working directory to: {work_directory}")

        try:
            path = parameters.file_path
            content = parameters.content

            # Check if the path is in the subdirectories of the current directory
            if not is_path_in_current_or_subdirectory(path):
                logger.error(f"Path {path} is not in the current or subdirectory")
                return WriteFileResult(
                    path=path,
                    status="ERROR: Path is not in the current or subdirectory",
                )

            # Log if the file already exists and will be overwritten
            if os.path.isfile(path):
                logger.warning(f"File {path} already exists and will be overwritten")

            # create the directory if it does not exist
            os.makedirs(os.path.dirname(path), exist_ok=True)

            # Write the content to the file
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(content))

            # Check if the file was written correctly by checking its size
            file_size = os.path.getsize(path)
            logger.info(f"File {path} written with size {file_size}")

            # report the size of the file written
            status = "OK"

            if file_size == 0 and (content != [] or content != [""]):
                logger.warning(f"File {path} written with size 0, content: {content}")
                return WriteFileResult(
                    path=path, status="ERROR: File written with size 0"
                )

        except OSError as e:
            logger.error(f"Error writing file: {e}")
            status = f"ERROR: {e}"

        final_result = WriteFileResult(path=path, status=status)
        return final_result


if __name__ == "__main__":

    # Create an instance of the WriteFiles tool
    file_writer = WriteFile()

    # Define the input parameters for the WriteFiles tool
    params = WriteFileParams(file_path="./output.txt", content=["Hello, world!\n"])

    # Run the WriteFiles tool with the input parameters
    output = file_writer.run(params)

    # Print the output of the WriteFiles tool
    logger.info(output)
