"""
Run Python code in a Docker container using the Docker SDK for Python.
"""

import os
import tarfile
import tempfile
from pathlib import Path
from typing import List

import docker
import docker.errors
from docker.models.containers import Container
from loguru import logger
from pydantic import Field

from elemental_agents.core.toolbox.tool import (
    Tool,
    ToolInitParameters,
    ToolParameters,
    ToolResult,
)
from elemental_agents.utils.exceptions import ToolException


class PythonRunnerParams(ToolParameters):
    """
    Parameters for the PythonRunner tool.

    :param filenames: A list of Python files to run.
    :param command: The command to run inside the container.
    :param requirements: A list of Python packages to install.
    """

    filenames: List[str] = Field(
        ...,
        description=(
            "A complete list of Python filenames needed to execute the application. "
            "Files will be copied to the new container which does not preserve previously "
            "stored data. Every needed file needs to be in this list."
        ),
    )
    command: str = Field(..., description="The command to run inside the container")
    requirements: List[str] = Field(
        default=[],
        description=(
            "A list of Python packages to install "
            "to fullfil the requirements of the application."
        ),
    )
    output_files: List[str] = Field(
        default=[],
        description=(
            "A list of files to copy from the container to the host. "
            "These files are the output of the application."
        ),
    )


class PythonRunnerResult(ToolResult):
    """
    Result for the PythonRunner tool.

    :param output: The output of the Python code execution.
    """

    output: str = Field(..., description="The output of the Python code execution")

    def __str__(self) -> str:
        return f"{self.output}"


class PythonRunnerInitParameters(ToolInitParameters):
    """
    Initialization parameters for the PythonRunner tool.
    """

    base_image: str = Field(
        default="python:3.10-slim",
        description="The base Docker image to use for the container",
    )


class PythonRunner(Tool):
    """
    Tool for running Python code in a Docker container.
    """

    name = "PythonRunner"
    description = (
        "Run Python code in a stateless Docker container. "
        " Files are copied to the empty container, requirements "
        "installed and your command is run. The standard output is "
        "returned. Any created files need to be copied as output files "
        "to keep them. Container is then removed."
    )
    example = {
        "name": "PythonRunner",
        "parameters": {
            "filenames": ["main.py", "utils.py"],
            "command": "python main.py",
            "requirements": ["requests"],
            "output_files": ["output.txt"],
        },
    }

    def run(self, parameters: PythonRunnerParams) -> PythonRunnerResult:  # type: ignore
        """
        Run Python code in a Docker container.

        :param parameters: The parameters for the PythonRunner tool.
        :return: The output of the Python code execution.
        """

        atomic_workspace = os.getenv("ATOMIC_WORKSPACE", "./")
        work_directory = Path(atomic_workspace).resolve()  # Resolve absolute path

        # Change the current working directory to ATOMIC_WORKSPACE
        os.chdir(work_directory)
        logger.info(f"Changed working directory to: {work_directory}")

        filenames = parameters.filenames
        command = parameters.command
        requirements = parameters.requirements

        session = DockerPythonSession()
        result = session.start_session(filenames, requirements)
        if result != "Session started successfully.":
            session.stop_session()
            logger.error(f"Error starting session: {result}")
            raise ToolException(f"Error starting session: {result}")

        output = session.execute_command(command)

        if parameters.output_files:
            output_files = parameters.output_files
            output_path = work_directory

            session.copy_files_from_container(output_files, str(output_path))

        session.stop_session()
        return PythonRunnerResult(output=output)


class DockerPythonSession:
    """
    Manages a Python session inside a Docker container.
    """

    def __init__(self, base_image: str = "python:3.10-slim") -> None:
        """
        Initializes the DockerPythonSession object.

        :param base_image: The base Docker image to use for the container.
        """

        self.client = docker.from_env()
        self.base_image = base_image
        self.container: Container = None
        self.temp_dir = tempfile.mkdtemp()
        self.installed_requirements: set[str] = set()

    def _copy_files_to_temp(self, filenames: List[str]) -> None:
        """
        Copies the specified files to the temporary directory.

        :param filenames: A list of filenames to copy.
        """
        for filename in filenames:
            if not os.path.exists(filename):
                raise ToolException(f"File {filename} does not exist.")
            dest_path = os.path.join(self.temp_dir, os.path.basename(filename))
            with (
                open(filename, "r", encoding="utf-8") as src,
                open(dest_path, "w", encoding="utf-8") as dest,
            ):
                dest.write(src.read())

    def _install_requirements(self, full_requirements: List[str]) -> str:
        """
        Installs the specified Python packages in the Docker container.

        :param full_requirements: A list of Python packages to install.
        :return: A message indicating whether the packages were installed successfully.
        """

        if self.container is not None:
            requirements_to_install = (
                set(full_requirements) - self.installed_requirements
            )
            if requirements_to_install:
                install_cmd = (
                    f"pip install --no-cache-dir {' '.join(requirements_to_install)}"
                )
                exec_result = self.container.exec_run(install_cmd)
                output = exec_result.output.decode("utf-8")
                if exec_result.exit_code == 0:
                    self.installed_requirements.update(requirements_to_install)
                else:
                    return f"Error installing packages: {output}"
            return "Successfully installed requirements."
        return "Error: No container session is running."

    def start_session(
        self, filenames: List[str], requirements: List[str] = None
    ) -> str:
        """
        Starts a Docker container and copies the specified files to it.

        :param filenames: A list of filenames to copy to the container.
        :param requirements: A list of Python packages to install.
        :return: A message indicating whether the session started successfully.
        """

        self._copy_files_to_temp(filenames)

        dockerfile = f"FROM {self.base_image}\n"
        dockerfile += "COPY . /app\n"
        dockerfile += "WORKDIR /app\n"

        dockerfile_path = os.path.join(self.temp_dir, "Dockerfile")
        with open(dockerfile_path, "w", encoding="utf-8") as f:
            f.write(dockerfile)

        try:
            image, _ = self.client.images.build(
                path=self.temp_dir, tag="python-app-runner-session", rm=True
            )
        except docker.errors.BuildError as e:
            return f"Error building image: {str(e)}"
        except docker.errors.DockerException as e:
            return f"Docker error during building the image: {str(e)}"

        try:
            self.container = self.client.containers.run(
                image=image.id, detach=True, tty=True, stdin_open=True
            )
        except docker.errors.APIError as e:
            return f"API error occurred: {e}"
        except docker.errors.DockerException as e:
            return f"An error occurred with Docker: {e}"

        if requirements:
            try:
                status = self._install_requirements(requirements)

            except docker.errors.APIError as e:
                return f"API error occurred: {e}"
            except docker.errors.DockerException as e:
                return f"An error occurred with Docker: {e}"

            logger.debug(f"Python requirements in Docker : {status}")
            if status != "Successfully installed requirements.":
                return status

        return "Session started successfully."

    def execute_command(self, command: str) -> str:
        """
        Executes a command inside the Docker container.

        :param command: The command to execute.
        :return: The output of the command
        """

        if not self.container:
            return "Error: No container session is running."
        try:
            exec_result = self.container.exec_run(command)
            output = exec_result.output.decode("utf-8")
            return (
                output
                if exec_result.exit_code == 0
                else f"Error executing command: {output}"
            )
        except docker.errors.APIError as e:
            return f"API error occurred: {e}"
        except docker.errors.DockerException as e:
            return f"An error occurred with Docker: {e}"

    def copy_files_from_container(
        self, output_files: List[str], host_path: str
    ) -> None:
        """
        Copies specific files from the Docker container to the host directly.

        :param output_files: A list of specific files to copy from the container.
        :param host_path: The path on the host to copy files to.
        """

        if not self.container:
            raise ToolException("Error: No container session is running.")

        for file_name in output_files:
            container_file_path = (
                f"/app/{file_name}"  # Use the container's working directory
            )
            host_file_path = os.path.join(host_path, file_name)

            try:
                # Retrieve the file content from the container
                bits, _ = self.container.get_archive(container_file_path)

                # Create a temporary tar file
                temp_tar = os.path.join(self.temp_dir, f"{file_name}.tar")
                with open(temp_tar, "wb") as f:
                    for chunk in bits:
                        f.write(chunk)

                # Extract the file from the tar
                with tarfile.open(temp_tar) as tar:
                    tar.extractall(path=host_path, filter="data")

                # Rename the extracted file if needed
                extracted_name = os.path.basename(container_file_path)
                if extracted_name != file_name:
                    os.rename(os.path.join(host_path, extracted_name), host_file_path)

            except docker.errors.NotFound as e:
                logger.error(
                    f"File {file_name} not found in container at {container_file_path}"
                )
                raise ToolException(f"File {file_name} not found in container") from e
            except docker.errors.APIError as e:
                logger.error(f"Error copying file {file_name} from container: {e}")
                raise ToolException(
                    f"Error copying file {file_name} from container"
                ) from e
            except Exception as e:
                logger.error(f"Error copying file {file_name} from container: {e}")
                raise ToolException(
                    f"Error copying file {file_name} from container"
                ) from e

    def stop_session(self) -> None:
        """
        Stops the Docker container and removes the temporary directory.
        """

        if self.container:
            self.container.stop()
            self.container.remove()
        self.container = None
        for file_name in os.listdir(self.temp_dir):
            os.remove(os.path.join(self.temp_dir, file_name))
        os.rmdir(self.temp_dir)


if __name__ == "__main__":

    files = ["integrated_component.py"]
    COMMAND = "python integrated_component.py"
    my_requirements: List[str] = []
    my_output_files = ["output.svg"]
    params = PythonRunnerParams(
        filenames=files,
        command=COMMAND,
        requirements=my_requirements,
        output_files=my_output_files,
    )
    tool = PythonRunner()
    my_result = tool.run(params)
    logger.info(my_result)
