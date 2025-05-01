"""
Main command line interface to run the workflow with the specified configuration
file and input task.
"""

from pathlib import Path

# pylint: disable=no-value-for-parameter
import click
from loguru import logger
from rich.console import Console

from elemental_agents.core.driver.driver import Driver
from elemental_agents.utils.utils import get_random_string


@click.command()
@click.option(
    "--config", default="config.yaml", help="Path to the YAML configuration file."
)
@click.option("--instruction", default=None, help="Input task to run.")
def main(config: str, instruction: str) -> None:
    """
    Main function to run the workflow with the specified configuration file.
    """

    console = Console()

    # Warn that default configuration file is being used
    if config == "config.yaml":
        logger.warning("Using the default configuration file: config.yaml")

    # Warn the user if the configuration file is not found
    if not Path(config).exists():
        logger.error(f"Configuration file not found: {config}")
        raise FileNotFoundError(f"Configuration file not found: {config}")

    # Load the configuration and print the workflow config
    elemental_driver = Driver(config)
    elemental_driver.load_config()

    logger.info("Configuration loaded.")
    console.print(elemental_driver.configuration())

    # Setup the workflow
    elemental_driver.setup()

    # Run the workflow
    input_session_id = get_random_string(10)
    output = elemental_driver.run(instruction, input_session_id)

    console.print(output)
    logger.info(f"Workflow from {config} completed.")


if __name__ == "__main__":

    main()
