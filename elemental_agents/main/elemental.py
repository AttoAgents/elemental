"""
Main command line interface to run the workflow with the specified configuration
file and input task.
"""

# pylint: disable=no-value-for-parameter

import asyncio
import json
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict
from urllib.parse import urlparse

import click
import socketio
from aiohttp import web
from loguru import logger
from rich import get_console
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown

from elemental_agents.core.driver.driver import Driver
from elemental_agents.core.memory.short_memory import ShortMemory
from elemental_agents.llm.data_model import Message
from elemental_agents.utils.config import ConfigModel
from elemental_agents.utils.utils import get_random_string

# Global variables for signal handling
stop_event = threading.Event()


def signal_handler(sig: int, frame: Any) -> None:
    """
    Handle Ctrl+C to gracefully exit.
    """
    print("\nStopping...")
    stop_event.set()
    sys.exit(0)


# Set up signal handler for graceful exit
signal.signal(signal.SIGINT, signal_handler)


async def start_socket_relay_server(socket_url: str) -> None:
    """
    Start the Socket.IO relay server.

    :param socket_url: URL of the Socket.IO relay server
    """

    parsed_url = urlparse(socket_url)
    ip = parsed_url.hostname
    port = parsed_url.port

    # Create a Socket.IO server
    sio = socketio.AsyncServer(cors_allowed_origins=socket_url, async_mode="aiohttp")

    app = web.Application()
    sio.attach(app)

    @sio.event
    async def connect(sid: str, environ: Any) -> None:
        """
        Handle client connection.

        :param sid: Socket ID of the client
        :param environ: Environment variables
        """
        logger.debug(f"Client connected: {sid}")

    @sio.event
    async def disconnect(sid: str) -> None:
        """
        Handle client disconnection.

        :param sid: Socket ID of the client
        """
        logger.debug(f"Client disconnected: {sid}")

    @sio.event
    async def message(sid: str, data: str) -> Dict[str, str]:
        """
        Handle generic messages from clients by broadcasting
        them to all connected clients.

        :param sid: Socket ID of the client
        :param data: Message data
        :return: Acknowledgment message
        """

        # Broadcast to all clients
        await sio.emit("message", data)

        return {"status": "received"}

    logger.debug(f"Starting Socket.IO server at {ip}:{port}...")
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, ip, port)
    await site.start()

    # Keep the server running until stop_event is set
    while not stop_event.is_set():
        await asyncio.sleep(1)

    # Clean up
    logger.debug("Shutting down Socket.IO server...")
    await runner.cleanup()
    logger.debug("Socket.IO server shut down")


def run_socket_relay_server(socket_url: str) -> None:
    """
    Run the Socket.IO relay server in a separate thread. LLM module will act as
    a client to this server.
    """
    asyncio.run(start_socket_relay_server(socket_url))


def run_display(socket_url: str, complete_event: threading.Event) -> None:
    """
    Run the display that connects to the Socket.IO relay server.
    Closes when complete_event is set or when a message with 'full_response' is received.

    :param socket_url: URL of the Socket.IO relay server
    :param complete_event: Event to signal when response is complete
    """
    # Create a Socket.IO client
    sio = socketio.Client()
    console = Console()
    accumulated_content = ""

    @sio.event
    def connect() -> None:
        logger.debug("Display connected to Socket.IO relay server")

    @sio.event
    def disconnect() -> None:
        logger.debug("Display disconnected from Socket.IO relay server")

    @sio.event
    def message(data: str) -> None:
        nonlocal accumulated_content

        try:
            # Process different message formats
            if isinstance(data, dict):
                # Direct dictionary
                if "content" in data:
                    accumulated_content += data["content"]
                # Check for completion signal
                if "full_response" in data:
                    logger.debug("Received full_response signal")
                    complete_event.set()

            elif isinstance(data, str):
                # JSON string
                try:
                    json_data = json.loads(data)
                    if "content" in json_data:
                        accumulated_content += json_data["content"]
                    # Check for completion signal
                    if "full_response" in json_data:
                        logger.debug("Received full_response.")
                        accumulated_content = ""
                        # complete_event.set()
                except json.JSONDecodeError:
                    # Not JSON, treat as plain text
                    accumulated_content += data
            else:
                # Unknown format
                accumulated_content += f"\nReceived: {json.dumps(data)}\n"

        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")

    # Connect to the Socket.IO relay server
    try:
        sio.connect(socket_url)
    except Exception as e:
        logger.error(f"Error connecting to Socket.IO relay server: {str(e)}")
        return

    # Live display for streaming content using Rich library
    live = None
    try:
        with Live(
            Markdown(accumulated_content),
            refresh_per_second=10,
            console=console,
            vertical_overflow="visible",
        ) as live:
            # Run until stop event is set or full_response is received
            while not stop_event.is_set() and not complete_event.is_set():
                live.update(Markdown(accumulated_content))
                time.sleep(0.1)

            # # If we received full_response, give a moment to display final content
            # if complete_event.is_set():
            #     if accumulated_content != "":

            #         live.update(Markdown(accumulated_content))
            #         time.sleep(0.5)  # Short delay to ensure final content is visible

    except Exception as e:
        logger.error(f"Error in display: {str(e)}")
    finally:
        # Ensure the Live display is properly closed
        if live and hasattr(live, "_stop_live"):
            try:
                live._stop_live()
            except Exception as e:
                logger.debug(f"Error stopping live display: {str(e)}")

        # Disconnect from the Socket.IO relay server
        if sio.connected:
            try:
                sio.disconnect()
            except Exception as e:
                logger.debug(f"Error disconnecting socket: {str(e)}")

        logger.debug("Display thread completed")


def set_debug_mode(debug: bool) -> None:
    """
    Set the debug mode for logging.

    :param debug: Boolean flag to enable or disable debug mode
    """
    if debug:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")
        logger.debug("Debug mode enabled.")
    else:
        logger.remove()
        logger.add(sys.stderr, level="INFO")


def blue_prompt(message: str = "(type 'exit' to quit)", console: Console = None) -> str:
    """
    Display a blue prompt and get user input directly.

    :param message: Message to display before the prompt
    :param console: Console object for printing
    :return: User input from the prompt
    """
    console = console or get_console()

    # Print the message
    console.print("")
    console.print(message, end="")

    # Force a new line and flush
    print("", flush=True)

    # Print the blue prompt
    console.print("[bold blue]>>>[/bold blue] ", end="")

    # Get input directly
    return input()


@click.command()
@click.option(
    "--config", default="config.yaml", help="Path to the YAML configuration file."
)
@click.option("--instruction", default=None, help="Input task to run.")
@click.option("--debug", is_flag=True, help="Enable debug mode.")
@click.option("--save", is_flag=True, help="Save the conversation to a file.")
@click.option(
    "--file", default=None, help="Name of the JSON file to save the conversation."
)
def main(
    config: str,
    instruction: str = None,
    debug: bool = False,
    save: bool = False,
    file: str = None,
) -> None:
    """
    Main function to run the workflow with the specified configuration file.

    :param config: Path to the YAML configuration file
    :param instruction: Input task to run
    :param debug: Enable debug mode
    :param save: Save the conversation to a file
    :param file: File to save the conversation
    """
    # Set debug mode if specified
    set_debug_mode(debug)

    settings = ConfigModel()
    socket_url = settings.websocket_url
    default_short_memory = settings.short_memory_items
    logger.debug(f"Socket URL: {socket_url}")

    global stop_event
    stop_event.clear()  # Reset the stop event

    console = Console()

    # Warn that default configuration file is being used
    if config == "config.yaml":
        logger.warning("Using the default configuration file: config.yaml")

    # Warn the user if the configuration file is not found
    if not Path(config).exists():
        logger.error(f"Configuration file not found: {config}")
        raise FileNotFoundError(f"Configuration file not found: {config}")

    # Check if filename is provided if save is enabled
    if save and file is None:
        logger.error(
            (
                "Filename must be provided if save (--save) is enabled. "
                "Use --file <filename>."
            )
        )
        raise ValueError("Filename must be provided if save is enabled.")

    # Start the Socket.IO relay server in a separate thread
    relay_thread = threading.Thread(target=run_socket_relay_server, args=(socket_url,))
    relay_thread.daemon = True
    relay_thread.start()

    # Give the relay server time to start
    time.sleep(1)

    # Define the display in a separate thread
    display_thread = threading.Thread(target=run_display, args=(socket_url,))

    try:
        # Load the configuration and print the workflow config
        elemental_driver = Driver(config)
        elemental_driver.load_config()

        logger.info("Configuration loaded.")
        config_info = elemental_driver.configuration()
        console.print(config_info)

        # Setup the workflow
        elemental_driver.setup()

        # Run the workflow in the main thread
        input_session_id = get_random_string(10)

        # If no instruction is provided, prompt the user
        # for input and enter interactive loop
        if instruction is None:

            # Short memory will keep everything to save history later
            sm = ShortMemory(capacity=default_short_memory)

            console.print("No instruction provided. Enter your task:")
            while not stop_event.is_set():
                try:
                    # Read input from the user
                    instruction = blue_prompt(console=console)
                    if instruction.lower() == "exit":
                        break

                    sm.add(Message(role="user", content=instruction))

                    # Create a new response_complete event for this iteration
                    response_complete = threading.Event()

                    # Clean up previous display thread if it exists
                    if display_thread and display_thread.is_alive():
                        display_thread.join(timeout=2)

                    # Create and start a new display thread
                    display_thread = threading.Thread(
                        target=run_display, args=(socket_url, response_complete)
                    )
                    display_thread.daemon = True
                    display_thread.start()

                    time.sleep(0.5)  # Give display thread time to connect

                    # Run the workflow with the provided instruction
                    agent_instruction = [msg.content for msg in sm.get_all()]

                    output = elemental_driver.run(agent_instruction, input_session_id)

                    # add the output to short memory
                    sm.add(Message(role="assistant", content=output))

                    response_complete.set()  # Safeguard in case of non-streaming response

                    # Wait for display thread to finish
                    if display_thread and display_thread.is_alive():
                        display_thread.join(timeout=2)

                    # Reset terminal colors again before printing output
                    console.clear_live()
                    print("\033[0m", end="", flush=True)

                    console.print(Markdown(output))
                except EOFError:
                    break
                except KeyboardInterrupt:
                    break
        else:
            # Run the workflow with the provided instruction
            logger.debug(f"Running workflow with instruction: {instruction}")
            output = elemental_driver.run(instruction, input_session_id)

        logger.debug(f"Workflow from {config} completed.")

        # If save is enabled, save the conversation to a file
        if save:
            content = json.dumps([msg.model_dump() for msg in sm.get_all()])
            with open(file, "w", encoding="utf-8") as f:
                f.write(content)
            logger.info(f"Conversation saved to {file}")

        # Wait a moment for final messages to be displayed
        time.sleep(2)
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")

    finally:
        # Signal threads to stop
        stop_event.set()

        # Wait for threads to finish
        relay_thread.join(timeout=5)


if __name__ == "__main__":

    main()
