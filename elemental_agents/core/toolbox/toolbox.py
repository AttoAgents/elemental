"""
Implementation of the ToolBox class that allows registration of tools and their
execution. The ToolBox class provides a standardized way to call tools by name
and pass parameters to them.
"""

import asyncio
import importlib.util
import inspect
import json
import os
from pathlib import Path
from typing import Dict, Tuple

from loguru import logger
from mcp.types import CallToolResult as MCPToolResult
from mcp.types import Tool as MCPTool
from pydantic import BaseModel, ValidationError
from rich.console import Console

from elemental_agents.core.toolbox.mcp_client import MCPClientToolbox
from elemental_agents.core.toolbox.tool import (
    Tool,
    ToolInitParameters,
    ToolParameters,
    ToolResult,
)


class ErrorResult(ToolResult):
    """
    ErrorResult class to represent an error result from a tool execution.
    """

    error: str = None

    def __str__(self) -> str:
        """
        String representation of the error result.

        :return: String representation of the error result
        """
        return f"Error: {self.error}"


class ToolBox:
    """
    Toolbox class to manage tools and their execution by the agent. Toolbox
    allows the standardized execution of tools that are registered in the toolbox
    and invocation of the tools by name from parsed JSON parameters.
    """

    def __init__(self) -> None:
        """
        Constructor, creates an empty tool registry.
        """

        self._tool_registry: Dict[str, Tuple[BaseModel, BaseModel, BaseModel]] = {}
        self._mcp_client_registry: Dict[str, MCPClientToolbox] = {}
        self._mcp_tool_registry: Dict[str, MCPTool] = {}

    def register_tool(
        self,
        tool_name: str,
        tool_class: BaseModel,
        param_model_class: BaseModel,
        init_model_class: BaseModel = None,
    ) -> None:
        """
        Register a tool with the toolbox.

        :param tool_name: Name of the tool
        :param tool_class: Tool class
        :param param_model_class: Pydantic model class for the tool's parameters
        :param init_model_class: Pydantic model class for the tool's
            initialization parameters (optional)
        """

        self._tool_registry[tool_name] = (
            tool_class,
            param_model_class,
            init_model_class,
        )

    def call_tool(
        self, tool_name: str, parameters_json: str
    ) -> ToolResult | MCPToolResult:
        """
        Call a tool by name with the specified parameters provided as a JSON string.

        :param tool_name: Name of the tool to call
        :param parameters_json: JSON string with parameters for the tool
        :return: Result of the tool execution
        """

        if "MCP|" in tool_name:
            return self.call_mcp_tool(tool_name, parameters_json)

        try:
            # Get the tool class and the associated Pydantic model for parameters
            tool_class, param_model_class, init_model_class = self._tool_registry.get(
                tool_name, (None, None, None)
            )

            if tool_class is None or param_model_class is None:
                raise ValueError(f"Tool '{tool_name}' not found.")

            # Parse JSON parameters into the Pydantic model
            parameters_data = json.loads(parameters_json)
            validated_params = param_model_class(**parameters_data)  # type: ignore

            # Instantiate the tool and call the run method
            tool_instance = tool_class(init_model_class)  # type: ignore
            result = tool_instance.run(validated_params)

            return result

        except ValidationError as e:
            logger.error(f"Validation Error: {e}")

        except (ValueError, TypeError) as e:
            logger.error(f"Error: {e}")

        return ToolResult()

    def call_mcp_tool(
        self, tool_name: str, parameters_json: str
    ) -> ToolResult | MCPToolResult:
        """
        Call an MCP tool by name with the specified parameters provided as a JSON string.

        :param tool_name: Name of the tool to call
        :param parameters_json: JSON string with parameters for the tool
        :return: Result of the tool execution
        """

        try:
            # Get the MCP server
            mcp_server_name = tool_name.split("|")[1]
            mcp_tool_name = tool_name.split("|")[2]

            mcp_server = self._mcp_client_registry.get(mcp_server_name)
            if mcp_server is None:
                raise ValueError(f"MCP Server '{mcp_server_name}' not found.")

            parameters = json.loads(parameters_json)
            coroutine = mcp_server.run_tool(mcp_tool_name, parameters)
            result = asyncio.run(coroutine)  # type: ignore

            if result is None:
                raise ValueError(
                    f"Tool '{mcp_tool_name}' on '{mcp_server_name}' server returned None"
                )

            return result

        except ValidationError as e:
            logger.error(f"Validation Error: {e}")
            return ErrorResult(error=str(e))

        except (ValueError, TypeError) as e:
            logger.error(f"Error: {e}")
            return ErrorResult(error=str(e))

    def describe(self) -> Dict[str, str]:
        """
        Get descriptions of all registered tools in a dictionary with name of
        the tool as key and description as value. Example description:

        { "Calculator": "Performs basic arithmetic, Parameters: {"expression":
        {"type": "string", "default": null}}",
          "ListFiles": "Lists files in a directory, Parameters: {"path": {"type": "string",
        "default": null}, "files_only": {"type": "boolean", "default":
        null}}" }

        :return: Dictionary with tool names and descriptions
        """

        tool_descriptions = {}

        for tool_name, (tool_class, tool_params, _) in self._tool_registry.items():
            parameter_descriptions = {}
            params = tool_params.model_json_schema()["properties"]
            for key, value in params.items():
                if "default" not in value:
                    value["default"] = None

                parameter_descriptions[key] = {
                    "description": value.get("description", ""),
                    "type": value.get("type", ""),
                    "default": value.get("default", None),
                }

            # Serialize parameter descriptions as JSON string with consistent double quotes
            string_descriptions = json.dumps(parameter_descriptions)

            # Serialize tool description as JSON string to ensure consistent quoting
            tool_description = json.dumps(tool_class.get_description())  # type: ignore

            description_str = (
                f"Description: {tool_description}, Parameters: {string_descriptions}"
            )

            if tool_class.get_example():  # type: ignore
                # Append example as JSON string if it is not already a string
                example = tool_class.get_example()  # type: ignore
                # If example is a dict or list, serialize it; else keep as is
                if not isinstance(example, str):
                    example = json.dumps(example)
                description_str += f", Example: {example}"

            tool_descriptions[tool_name] = description_str

        # Add MCP tools descriptions
        for tool_name, tool in self._mcp_tool_registry.items():
            # Serialize description and inputSchema as JSON strings for consistency
            description_json = json.dumps(tool.description)
            input_schema_json = json.dumps(tool.inputSchema)
            tool_descriptions[tool_name] = (
                f"Description: {description_json}, Parameters: {input_schema_json} "
            )

        return tool_descriptions

    def describe_short(self) -> str:
        """
        Get descriptions of all registered tools in a dictionary with name of
        the tool as key and description as value. Example description:

        "Performs basic arithmetic; Lists files in a directory"

        :return: String all tool descriptions (aka skills of the agent)
        """

        tool_descriptions = ""

        for _, (tool_class, _, _) in self._tool_registry.items():

            tool_descriptions += f"{tool_class.get_description()}; "  # type: ignore

        # Add MCP tools descriptions
        for _, tool in self._mcp_tool_registry.items():
            tool_descriptions += f"{tool.description}; "

        return tool_descriptions

    def discover_tools(self) -> Dict[str, Tuple]:
        """
        Discover tools from multiple locations and return a dictionary of tools.

        :return: A dictionary mapping tool names to their classes and parameters.
        """
        tool_directories = []

        # Default tools directory (relative to the script location)
        base_directory = Path(__file__).parent / "../../tools"
        logger.info(f"Tools :: Base directory: {base_directory}")
        if base_directory.exists() and base_directory.is_dir():
            tool_directories.append(base_directory)

        # Environment variable directories (system-wide and user-specific)
        system_tools_dir = os.getenv("ATTO_SYSTEM_TOOLS_DIR")
        user_tools_dir = os.getenv("ATTO_USER_TOOLS_DIR")

        for env_dir in [system_tools_dir, user_tools_dir]:
            if env_dir:  # Ensure it's not None
                path = (
                    Path(env_dir).expanduser().resolve()
                )  # Expand `~` and resolve path
                logger.info(f"Tools :: Environment directory: {path}")
                if path.exists() and path.is_dir():
                    tool_directories.append(path)

        # Discover tools from all valid directories
        tools_dict = {}
        for directory in tool_directories:
            tools_dict.update(self.load_classes_from_files(directory))

        logger.debug(f"Discovered tools: {tools_dict.keys()}")

        return tools_dict

    def load_classes_from_files(self, directory: Path) -> Dict[str, tuple]:
        """
        Load classes from Python files in a directory and return a dictionary of tools.

        :param directory: The directory containing the Python files
        :return: A dictionary mapping tool names to their classes and parameters
        """
        tools_dict = {}

        for filename in os.listdir(directory):

            # logger.debug(f"Loading file: {filename}")

            if filename.endswith(".py"):
                module_name = filename[:-3]  # Remove the .py extension
                module_path = os.path.join(directory, filename)

                # Load the module
                spec = importlib.util.spec_from_file_location(module_name, module_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                # Dynamically search for classes based on suffixes
                tool_class = None
                tool_params_class = None
                tool_init_params_class = None

                # Iterate through all classes in the module
                for _, cls in inspect.getmembers(module, inspect.isclass):

                    # Find the tool class
                    if issubclass(cls, Tool) and (cls != Tool):
                        tool_class = cls
                        # logger.debug(f"Found tool class: {cls}")

                    # Find the tool parameters class
                    if issubclass(cls, ToolParameters) and (cls != ToolParameters):
                        tool_params_class = cls
                        # logger.debug(f"Found tool parameter class: {cls}")

                    # Find the init parameters class
                    if issubclass(cls, ToolInitParameters) and (
                        cls != ToolInitParameters
                    ):
                        tool_init_params_class = cls
                        # logger.debug(f"Found tool init parameter class: {cls}")

                # If we found a tool class and its associated parameters class, proceed
                if tool_class and tool_params_class:

                    # Create an instance of the tool class to access its name attribute
                    tool_instance = tool_class()
                    tool_name = tool_instance.name  # Assuming 'name' attribute exists
                    tools_dict[tool_name] = (
                        tool_class,
                        tool_params_class,
                        tool_init_params_class,
                    )

        return tools_dict

    def register_tool_by_name(self, tool_name: str) -> None:
        """
        Register a tool by name from the tool registry.

        :param tool_name: Name of the tool to register
        """

        # Check if the tool is an MCP tool
        if "MCP|" in tool_name:
            self.register_mcp_tool_by_name(tool_name)
            return

        available_tools = self.discover_tools()
        if tool_name in available_tools:
            selected_tool = available_tools[tool_name]

            tool_class = selected_tool[0]
            tool_params_class = selected_tool[1]
            tool_params_init_class = selected_tool[2]

            self.register_tool(
                tool_name, tool_class, tool_params_class, tool_params_init_class
            )
        else:
            logger.error(f"Tool {tool_name} not found.")

    def register_mcp_server_with_all_tools(self, mcp_server_name: str) -> None:
        """
        Register all tools from the MCP server.

        :param mcp_server_name: Name of the MCP server
        """

        logger.info(f"Registering all tools from MCP server: {mcp_server_name}")
        try:
            mcp_server = MCPClientToolbox(mcp_server_name)
            self._mcp_client_registry[mcp_server_name] = mcp_server

            # List all available tools at the server
            coroutine = mcp_server.list_tools()
            result = asyncio.run(coroutine)  # type: ignore
            if result is None:
                raise ValueError(
                    f"Cannot register tools from MCP server: {mcp_server_name}"
                )
            for tool in result:
                # Register the tool in the toolbox
                registry_name = f"MCP|{mcp_server_name}|{tool.name}"
                self._mcp_tool_registry[registry_name] = tool
                logger.info(
                    f"Registered MCP tool: {tool.name} from server: {mcp_server_name}"
                )

            logger.info(f"Registered all tools from MCP server: {mcp_server_name}")
        except ValueError as e:
            logger.error(f"Error registering MCP server: {e}")
            logger.error(f"Cannot register tools from server {mcp_server_name}")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            logger.error(f"Cannot register tools from server {mcp_server_name}")

    def register_mcp_tool_by_name(self, tool_name: str) -> None:
        """
        Register an MCP tool by name from the MCP client.

        :param tool_name: Name of the tool to register
        """

        mcp_server_name = tool_name.split("|")[1]
        mcp_tool_name = tool_name.split("|")[2]

        if mcp_tool_name == "*":
            self.register_mcp_server_with_all_tools(mcp_server_name)
            return

        logger.info(
            f"Trying to register MCP tool: {mcp_tool_name} "
            f"from server: {mcp_server_name} MCP server"
        )
        try:
            # Register the MCP server
            if mcp_server_name in self._mcp_client_registry:
                mcp_server = self._mcp_client_registry[mcp_server_name]
            else:
                mcp_server = MCPClientToolbox(mcp_server_name)
                self._mcp_client_registry[mcp_server_name] = mcp_server

            # Get the tool from the MCP server
            coroutine = self._mcp_client_registry[mcp_server_name].get_tool(
                mcp_tool_name
            )
            result = asyncio.run(coroutine)  # type: ignore
            if result is None or result.name != mcp_tool_name:
                raise ValueError(
                    f"Tool '{mcp_tool_name}' not found on server '{mcp_server_name}'"
                )

            # List all available tools at the server
            self._mcp_client_registry[mcp_server_name].report_tools()

            # Register the tool in the toolbox
            registry_name = f"MCP|{mcp_server_name}|{mcp_tool_name}"
            self._mcp_tool_registry[registry_name] = result

            logger.info(
                f"Registered MCP tool: {mcp_tool_name} from server: {mcp_server_name}"
            )

        except ValueError as e:
            logger.error(f"Error registering MCP server: {e}")
            logger.error(
                f"Cannot register tool {tool_name} from server {mcp_server_name}"
            )


if __name__ == "__main__":
    from rich.pretty import pprint

    console = Console()

    box = ToolBox()

    box.register_tool_by_name("PythonRunner")
    box.register_tool_by_name("Calculator")
    box.register_tool_by_name("MCP|Github|list_issues")
    box.register_tool_by_name("MCP|Github|search_repositories")
    description = box.describe()
    # logger.info(description)
    # console.print_json(description)
    pprint(description)

    pprint(box.describe_short())

    CALCULATOR_JSON = '{"expression": "2 + 3 * 4"}'
    final_result = box.call_tool("Calculator", CALCULATOR_JSON)
    logger.info(final_result)

    # RUNNER_JSON = (
    #     '{"filenames": ["./main.py"], "command": "python main.py", "requirements": []}'
    # )
    # logger.info(f"PythonRunner JSON: {RUNNER_JSON}")
    # tool_result = box.call_tool("PythonRunner", RUNNER_JSON)
    # logger.info(f"PythonRunner result: {tool_result}")
