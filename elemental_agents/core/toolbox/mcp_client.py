"""
Model Context Protocol client for Elemental. This module provides MCP server
management for the Elemental toolbox.
"""

import json
from contextlib import AsyncExitStack
from typing import Awaitable, Callable, Dict, List, Optional, ParamSpec, TypeVar

from loguru import logger
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
from pydantic import BaseModel

from elemental_agents.utils.config import ConfigModel


class MCPServerParameters(BaseModel):
    """
    Model for MCP server parameters.
    """

    command: str
    args: List[str]
    env: Optional[dict] = None


P = ParamSpec("P")
R = TypeVar("R")


def manage_connection(
    func: Callable[P, Awaitable[R]],
) -> Callable[P, Awaitable[R]]:
    """
    Decorator to manage the connection to the MCP server.
    This decorator will connect to the server, execute the function, and
    then close the connection.

    :param func: The function to execute.
    :return: The result of the function execution.
    """

    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        self = args[0]

        # Check if already connected
        if not getattr(self, "_is_connected", False):
            await self.connect_to_server()  # type: ignore[attr-defined]
            should_close = True
        else:
            should_close = False

        try:
            result = await func(*args, **kwargs)
        finally:
            if should_close:
                await self.close()  # type: ignore[attr-defined]
        return result

    return wrapper


class MCPClientToolbox:
    """
    Model Context Protocol client for Elemental. This class provides MCP server
    management for the Elemental toolbox.
    """

    def __init__(self, mcp_server_name: str):
        """
        Initialize the MCP client.

        :param mcp_server_name (str): The name of the MCP server.
        """
        self._config = ConfigModel()

        self._mcp_server_name = mcp_server_name
        self._mcp_config: MCPServerParameters = self.get_parameters()
        self._mcp_tools: List[types.Tool] = []

        self._session: Optional[ClientSession] = None
        self._stdio = None
        self._write = None
        self._exit_stack = AsyncExitStack()

        self._is_connected = False

        logger.debug(f"Initialized MCP client with server: {self._mcp_server_name}")

    async def __aenter__(self) -> "MCPClientToolbox":
        """
        Async context manager entry.
        """
        await self.connect_to_server()
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Optional[BaseException],
    ) -> bool:
        """
        Async context manager exit.
        """
        await self.close()
        return False

    def get_parameters(self) -> MCPServerParameters:
        """
        Get the parameters for the MCP server from the environment variables.

        :return: MCPServerParameters object with server parameters.
        """
        if self._mcp_server_name not in self._config.mcpServers:
            logger.error(f"Server {self._mcp_server_name} not found in config")
            raise ValueError(f"Server {self._mcp_server_name} not found in config")

        mcp_configs = json.loads(self._config.mcpServers)
        server_config = mcp_configs[self._mcp_server_name]
        config = MCPServerParameters(**server_config)
        return config

    async def connect_to_server(self) -> None:
        """
        Connect to an MCP server
        """
        if self._is_connected:
            logger.warning("Already connected to MCP server")
            return

        try:
            command = self._mcp_config.command
            arguments = self._mcp_config.args
            environment = self._mcp_config.env

            server_params = StdioServerParameters(
                command=command, args=arguments, env=environment
            )

            stdio_transport = await self._exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            self._stdio, self._write = stdio_transport
            self._session = await self._exit_stack.enter_async_context(
                ClientSession(self._stdio, self._write)
            )

            await self._session.initialize()
            self._is_connected = True
            logger.debug(f"Connected to MCP server: {self._mcp_server_name}")

        except Exception as e:
            logger.error(f"Failed to connect to MCP server: {e}")
            await self.close()  # Clean up on connection failure
            raise

    async def close(self) -> None:
        """
        Close the MCP client connection with proper error handling.
        """
        if not self._is_connected:
            return

        try:
            if self._exit_stack:
                await self._exit_stack.aclose()
            logger.debug("Closed MCP client connection")
        except Exception as e:
            logger.error(f"Error closing MCP client connection: {e}")
        finally:
            # Reset connection state
            self._session = None
            self._stdio = None
            self._write = None
            self._is_connected = False

    @manage_connection
    async def list_tools(self) -> List[types.Tool]:
        """
        List available tools on the MCP server.

        :return: List of available tools.
        """
        if not self._session:
            raise RuntimeError("MCP client is not connected to a server")

        # List available tools
        response = await self._session.list_tools()
        self._mcp_tools = response.tools

        return self._mcp_tools

    @manage_connection
    async def get_tool(self, tool_name: str) -> Optional[types.Tool]:
        """
        Get a specific tool by name.

        :param tool_name: The name of the tool to retrieve.
        :return: The tool object if found, None otherwise.
        """
        if not self._session:
            raise RuntimeError("MCP client is not connected to a server")

        # List available tools
        response = await self._session.list_tools()
        self._mcp_tools = response.tools

        for tool in self._mcp_tools:
            if tool.name == tool_name:
                return tool

        return None

    @manage_connection
    async def run_tool(
        self, tool_name: str, tool_args: Dict[str, str]
    ) -> types.CallToolResult:
        """
        Run a tool with the given parameters.

        :param tool_name: The name of the tool to run.
        :param params: The parameters to pass to the tool
        :return: The result of the tool execution.
        """
        if not self._session:
            raise RuntimeError("MCP client is not connected to a server")

        # Execute tool call
        result = await self._session.call_tool(tool_name, tool_args)

        return result

    def report_tools(self) -> None:
        """
        Report the available tools to the logger.
        """
        if not self._mcp_tools:
            logger.warning("No tools available or list_tools() not called")
            return

        logger.info(f"Available tools on {self._mcp_server_name} server:")
        names = [tool.name for tool in self._mcp_tools]
        logger.info(f"Tools: {', '.join(names)}")
