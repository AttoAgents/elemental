"""
Tool to unpack archive files (ZIP, tar.gz) to a specified directory.
"""

import os
import tarfile
import zipfile
from pathlib import Path

from loguru import logger
from pydantic import Field

from elemental_agents.core.toolbox.tool import Tool, ToolParameters, ToolResult
from elemental_agents.utils.utils import is_path_in_current_or_subdirectory


class UnpackArchiveParams(ToolParameters):
    """
    Defines the input parameters for the UnpackArchive tool.

    :param archive_path: Path to the archive file to unpack
    :param extract_to: Directory where the contents should be extracted
    """

    archive_path: str = Field(..., description="Path to the archive file to unpack.")
    extract_to: str = Field(
        default="./extracted",
        description="Directory where the contents should be extracted.",
    )


class UnpackArchiveResult(ToolResult):
    """
    Defines the output result for the UnpackArchive tool.
    """

    success: bool = Field(
        default=False, description="Indicates if the archive was unpacked successfully."
    )
    message: str = Field(
        default="", description="Message regarding the unpacking process."
    )

    def __str__(self) -> str:
        return f"Success: {self.success}, Message: {self.message}"


class UnpackArchive(Tool):
    """
    Tool to unpack archive files (ZIP, tar.gz) to a specified directory.
    """

    name = "UnpackArchive"
    description = "Unpacks archive files (ZIP, tar.gz) to a specified directory."

    def run(self, parameters: UnpackArchiveParams) -> UnpackArchiveResult:  # type: ignore
        """
        Unpack the specified archive file to the given directory.

        :param parameters: UnpackArchiveParams object with the archive path and extract directory
        :return: UnpackArchiveResult indicating success or failure
        """

        atomic_workspace = os.getenv("ATOMIC_WORKSPACE", "./")
        work_directory = Path(atomic_workspace).resolve()  # Resolve absolute path

        # Change the current working directory to ATOMIC_WORKSPACE
        os.chdir(work_directory)
        logger.info(f"Changed working directory to: {work_directory}")

        archive_path = Path(parameters.archive_path)
        extract_to = Path(parameters.extract_to)

        try:
            if not is_path_in_current_or_subdirectory(str(archive_path)):
                logger.error(
                    f"Path {archive_path} is not in the current or subdirectory"
                )
                raise ValueError(
                    f"Path {archive_path} is not in the current or subdirectory"
                )

            if not is_path_in_current_or_subdirectory(str(extract_to)):
                logger.error(f"Path {extract_to} is not in the current or subdirectory")
                raise ValueError(
                    f"Path {extract_to} is not in the current or subdirectory"
                )

            # Ensure the extract directory exists
            extract_to.mkdir(parents=True, exist_ok=True)

            # Determine the archive type and extract accordingly
            if archive_path.suffix == ".zip":
                with zipfile.ZipFile(archive_path, "r") as zip_ref:
                    zip_ref.extractall(extract_to)
                logger.info(f"ZIP archive unpacked successfully to: {extract_to}")

            elif archive_path.suffix in [".tar", ".gz", ".tgz"]:
                with tarfile.open(archive_path, "r:*") as tar_ref:
                    tar_ref.extractall(extract_to)
                logger.info(f"Tar archive unpacked successfully to: {extract_to}")

            else:
                message = "Unsupported archive format."
                logger.error(message)
                result = UnpackArchiveResult(success=False, message=message)
                return result

            result = UnpackArchiveResult(
                success=True, message="Archive unpacked successfully."
            )
            return result

        except (zipfile.BadZipFile, tarfile.TarError, OSError) as e:
            logger.error(f"Error unpacking archive: {e}")
            result = UnpackArchiveResult(success=False, message=str(e))
            return result
