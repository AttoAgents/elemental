"""
Main driver to read a workflow configuration file and run the workflow.
"""

from typing import Any, Dict, List

import yaml

from elemental_agents.core.driver.workflow_config import Workflow
from elemental_agents.observability.observer import observer


class Driver:
    """
    Driver class to load the configuration and run the workflow defined in the configuration file.
    """

    def __init__(self, filename: str) -> None:
        """
        Main driver class to load the configuration and run the workflow.

        :param filename: The configuration file to load.
        """

        self._config_file = filename
        self._config: Dict[str, Any] = {}
        self._workflow: Workflow = None

    def configuration(self) -> Dict[str, str]:
        """
        Return the configuration of the workflow.
        """
        output = self._config
        return output

    def load_config(self) -> None:
        """
        Load the YAML configuration file.
        """
        with open(self._config_file, "r", encoding="utf-8") as file:
            self._config = yaml.safe_load(file)

    def setup(self) -> None:
        """
        Setup the workflow.
        """
        self._workflow = Workflow(self._config)

    def run(
        self, input_instruction: List[str] | str = None, input_session: str = None
    ) -> str:
        """
        Run workflow with a given input.

        :param input_instruction(s): The input to the workflow.
        :return: The output of the workflow.
        """

        fist_instruction = (
            input_instruction[0]
            if isinstance(input_instruction, list)
            else input_instruction
        )
        current_instruction = (
            input_instruction[-1]
            if isinstance(input_instruction, list)
            else input_instruction
        )

        observer.log_session(input_session, fist_instruction)
        observer.log_interaction(input_session, "user", current_instruction)

        output = self._workflow.run(input_instruction, input_session)

        observer.log_interaction(input_session, "system", output)

        return output
