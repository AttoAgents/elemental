"""
Status and task classes are defined in the task.py file. The Status class is an
Enum class that defines the status of a task. The Task class is a Pydantic
BaseModel class that represents a task in the task queue. It has attributes such
as id, description, result, status, dependencies, and context. The status
attribute is an instance of the Status class, which can be READY, BLOCKED, DONE,
FAILED, or IN_PROGRESS. The dependencies attribute is a list of task IDs that
this task depends on, and the context attribute is a dictionary to store the
results of the tasks that this task depends on.
"""

from enum import Enum
from typing import Dict, List

from pydantic import BaseModel


class Status(Enum):
    """
    Enum class for the status of a task. The status can be READY, BLOCKED,
    DONE, FAILED or IN_PROGRESS.
    """

    READY = "ready"
    BLOCKED = "blocked"
    DONE = "done"
    FAILED = "failed"
    IN_PROGRESS = "in_progress"

    def __str__(self) -> str:
        """
        Return the string representation of the status.
        """
        return self.value


class Task(BaseModel):
    """
    Task class to represent a task in the task queue. A task has an ID, a
    description, a result, a status, dependencies, and context. The status of
    a task can be READY, BLOCKED, DONE, FAILED or IN_PROGRESS. Dependencies are
    the IDs of the tasks that this task depends on. Context is a dictionary to
    store the results of the tasks that this task depends on.
    """

    id: str
    description: str
    result: str = ""
    status: Status = Status.BLOCKED
    dependencies: List[str]
    context: Dict[str, str] = {}
    origin: str = ""

    def add_to_context(self, identifier: str, information: str) -> None:
        """
        Add information to the context of this task.

        :param task_id: The ID of the task whose result is being added or
            identifier of the context message.
        :param result: The information to be added to the context.
        """
        self.context[identifier] = information
