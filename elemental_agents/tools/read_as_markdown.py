"""
Tool to read the content of abroad range of file formats and
convert it to Markdown giving a plain text.
"""

import os
from pathlib import Path
from typing import Dict, List

from loguru import logger
from markitdown import MarkItDown
from pydantic import Field

from elemental_agents.core.toolbox.tool import Tool, ToolParameters, ToolResult
from elemental_agents.utils.utils import is_path_in_current_or_subdirectory


# Define specific Pydantic model for ReadFiles input
class ReadFilesAsMarkdownParams(ToolParameters):
    """
    Defines the input parameters for the ReadFilesAsMarkdown tool.

    :param paths: List of file paths to read
    """

    paths: List[str] = Field(
        default=["./document.pdf"],
        description=(
            "List of file paths to read. Files can be in "
            "PDF/PowerPoint/Word/Excel/HTML/CSV/JSON/XML/ZIP format. "
            "Content will be converted to Markdown."
        ),
    )


class ReadFilesAsMarkdownResult(ToolResult):
    """
    Defines the output result for the ReadFiles tool.
    """

    contents: Dict[str, str] = Field(
        default={},
        description="Dictionary of file paths and their read contents in Markdown format.",
    )

    def __str__(self) -> str:
        return f"{self.contents}"


class ReadFilesAsMarkdown(Tool):
    """
    Tool to read the content of a file and convert it to Markdown.
    """

    name = "ReadFilesAsMarkdown"
    description = (
        "Reads PDF/PowerPoint/Word/Excel/HTML/CSV/JSON/XML/ZIP files "
        "and provides their content."
    )

    def run(
        self, parameters: ReadFilesAsMarkdownParams  # type: ignore
    ) -> ReadFilesAsMarkdownResult:
        """
        Read the content of the specified file. The files can be read only from
        subdirectories of the current directory. Only files are read, not
        directories. Files bigger than 10MB are not read. Files do not need to be
        plain text. All files will be converted to Markdown.

        :param parameters: ReadFilesAsMarkdownParams object with the paths
        :return: Dictionary of the strings with the content of the files read
            from specified paths
        """

        atomic_workspace = os.getenv("ATOMIC_WORKSPACE", "./")
        work_directory = Path(atomic_workspace).resolve()  # Resolve absolute path

        # Change the current working directory to ATOMIC_WORKSPACE
        os.chdir(work_directory)
        logger.info(f"Changed working directory to: {work_directory}")

        contents = {}
        md = MarkItDown()

        try:
            for path in parameters.paths:

                # Check if the path is in the subdirectories of the current directory
                if not is_path_in_current_or_subdirectory(path):
                    message = (
                        f"ERROR: Path {path} is not in the current or subdirectory"
                    )
                    logger.error(message)
                    contents[path] = message
                    continue

                # Check if the path is a file
                if os.path.isfile(path):

                    # Big files protection
                    file_size = os.path.getsize(path)
                    if file_size > 10000000:
                        message = (
                            f"ERROR: File {path} is too big to read, size {file_size}"
                        )
                        logger.error(message)
                        contents[path] = message
                        continue

                    # Read the file
                    try:
                        data = md.convert(path)
                        text = data.text_content
                        contents[path] = text
                    except (OSError, RuntimeError) as e:
                        message = f"ERROR: Error converting file to Markdown: {e}"
                        logger.error(message)
                        contents[path] = message
                else:
                    message = f"ERROR: Path {path} is not a file"
                    logger.error(message)
                    contents[path] = message

        except OSError as e:
            message = f"ERROR: Error reading file: {e}"
            logger.error(message)
            contents[path] = message

        result = ReadFilesAsMarkdownResult(contents=contents)
        return result
