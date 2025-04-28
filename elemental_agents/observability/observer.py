"""
Agent workflow observability class to log events for debugging and monitoring
purposes. Observer information logged to the database or file may also be used
to recover the state of the workflow in case of failure.
"""

from typing import Literal, Tuple

from loguru import logger
from rich.console import Console

from elemental_agents.core.taskqueue.task import Task
from elemental_agents.llm.data_model import Message
from elemental_agents.observability.observer_data_model import (
    ObserverInteraction,
    ObserverMessage,
    ObserverSession,
    ObserverTask,
)
from elemental_agents.observability.observer_database import ObserverDatabase
from elemental_agents.observability.observer_webhook import send_to_webhook
from elemental_agents.utils.config import ConfigModel
from elemental_agents.utils.singleton import singleton
from elemental_agents.utils.utils import get_random_string


@singleton
class AgentObserver:
    """
    AgentObserver class to log events for debugging and monitoring purposes.
    """

    def __init__(
        self,
        destination: Literal["screen", "file", "db", "webhook", "none"],
        file_name: str = "",
        database: str = "",
    ) -> None:
        """
        Initialize the Observer object.

        :param destination: Destination to log the events. Can be "screen",
            "file", "webhook" or "db".
        :param file_name: Name of the file to log the events. Required if
            destination is "file".
        :param database: Database connection string to log the events. Required
            if destination is "db".
        """

        self._id = get_random_string(10)
        self._dest = destination
        self._console = Console()

        logger.info(
            f"Observer initialized with ID {self._id} and logging to {self._dest}."
        )

        if self._dest == "file":
            if not file_name:
                logger.info("Observer file name not provided. Using default.")
                self._file_name = config.observer_file_name

        if self._dest == "db":
            self._db_connection_string = database
            self._db = ObserverDatabase(database)
            self._db.create()

        if self._dest == "webhook":
            self._webhook_url = config.observer_webhook_url

    def get_destination(self) -> Tuple[str, str]:
        """
        Get the destination of the observer.

        :return: Destination of the observer.
        """
        if self._dest == "file":
            return ("file", self._file_name)
        if self._dest == "db":
            return ("db", self._db_connection_string)
        if self._dest == "screen":
            return ("screen", "")
        if self._dest == "none":
            return ("none", "")
        if self._dest == "webhook":
            return ("webhook", self._webhook_url)

    def log_session(
        self, input_session: str, first_message: str, workflow_id: str = None
    ) -> None:
        """
        Log the session to the destination.

        :param input_session: Input session for the session.
        :param first_message: First message in the session.
        """

        workflow = workflow_id if workflow_id else "custom"

        record = ObserverSession(
            observer_id=self._id,
            session_id=input_session,
            first_message=first_message,
            workflow_id=workflow,
        )

        if self._dest == "screen":
            self._console.print(record)

        if self._dest == "file":
            with open(self._file_name, "a", encoding="utf-8") as file:
                file.write(str(record) + "\n")

        if self._dest == "db":
            self._db.upsert_session(record)

        if self._dest == "webhook":
            send_to_webhook(self._webhook_url, "session", record)

        if self._dest == "none":
            pass

    def log_interaction(self, input_session: str, role: str, message: str) -> None:
        """
        Log the interaction to the destination.

        :param input_session: Input session for the interaction.
        :param role: Role in the interaction.
        :param message: Message in the interaction.
        """

        record = ObserverInteraction(
            observer_id=self._id,
            session_id=input_session,
            role=role,
            message=message,
        )

        if self._dest == "screen":
            self._console.print(record)

        if self._dest == "file":
            with open(self._file_name, "a", encoding="utf-8") as file:
                file.write(str(record) + "\n")

        if self._dest == "db":
            self._db.add_interaction(record)

        if self._dest == "webhook":
            send_to_webhook(self._webhook_url, "interaction", record)

        if self._dest == "none":
            pass

    def log_message(
        self,
        input_session: str,
        message: Message,
        agent_name: str = "",
        task_description: str = "",
    ) -> None:
        """
        Log the message to the destination.

        :param input_session: Input session for the message.
        :param message: Message to be logged.
        :param agent_name: Name of the agent.
        :param task_description: Description of the task.
        """

        record = ObserverMessage(
            observer_id=self._id,
            session_id=input_session,
            role=message.role,
            text=message.content,
            agent=agent_name,
            task=task_description,
        )

        if self._dest == "screen":
            self._console.print(record)

        if self._dest == "file":
            with open(self._file_name, "a", encoding="utf-8") as file:
                file.write(str(record) + "\n")

        if self._dest == "db":
            self._db.add_message(record)

        if self._dest == "webhook":
            send_to_webhook(self._webhook_url, "message", record)

        if self._dest == "none":
            pass

    def log_task(self, input_session: str, task: Task) -> None:
        """
        Log the task to the destination.

        :param input_session: Input session for the task.
        :param task: Task to be logged.
        """

        logger.debug(f"Logging task: {task}")

        record = ObserverTask(
            observer_id=self._id,
            session_id=input_session,
            task_id=task.id,
            description=task.description,
            status=task.status,
            result=task.result,
            dependencies=task.dependencies,
            context=task.context,
            origin=task.origin,
        )

        if self._dest == "screen":
            self._console.print(record)

        if self._dest == "file":
            with open(self._file_name, "a", encoding="utf-8") as file:
                file.write(str(record) + "\n")

        if self._dest == "db":
            self._db.upsert_task(record)

        if self._dest == "webhook":
            send_to_webhook(self._webhook_url, "task", record)

        if self._dest == "none":
            pass


config = ConfigModel()

match config.observer_destination:
    case "file":
        observer = AgentObserver(
            destination="file", file_name=config.observer_file_name
        )
    case "db":
        CONNECTION_STRING = config.observer_database_connection_string
        observer = AgentObserver(
            destination=config.observer_destination, database=CONNECTION_STRING
        )
    case "screen":
        observer = AgentObserver(destination="screen")
    case "webhook":
        observer = AgentObserver(destination="webhook")
    case "none":
        observer = AgentObserver(destination="none")
    case _:
        observer = AgentObserver(destination=config.observer_destination)
