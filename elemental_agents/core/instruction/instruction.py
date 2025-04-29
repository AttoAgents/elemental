"""
Generic definition of the instruction passed to the agent or directly to language model.
"""

from typing import List, Optional

from elemental_agents.core.taskqueue.queue import TaskQueue
from elemental_agents.core.taskqueue.task import Task


class BasicInstruction:
    """
    Definition of the instruction which is done outside of the taskqueue (i.e.
    direct interaction with the agent).
    """

    def __init__(self, task: str, memories: List[str] | None = None) -> None:
        """
        Initialize the instruction with the task.

        :param task: The task to be executed - simple string description.
        :param memories: The relevant memories from the long
            memory to be used in the task execution.
        """

        self._message = task
        self._memories = memories

    def render(self) -> str:
        """
        Create full string representation of the instruction including task
        description.
        """

        msg = f"{self._message}"

        if (self._memories is not None) and (len(self._memories)) > 0:

            for item in self._memories:
                msg += f"\n<memory> {item} </memory>"

        return msg

    def __str__(self) -> str:
        return self.render()


class Instruction:
    """
    Definition of the instruction which in addition to task includes potential
    context and memories retrieved from the long memory.
    """

    def __init__(self, task: Task, memories: Optional[List[str]] = None) -> None:
        """
        Initialize the instruction with the task and optional memories.

        :param task: The task to be executed.
        :param memories: The relevant memories from the long
            memory to be used in the task execution.
        """

        self._description = task.description
        self._context = task.context
        self._memories = memories

    def render(self) -> str:
        """
        Create full string representation of the instruction including task
        description, context, and memories. Last two are optional.
        """

        full_instruction = f"<task>{self._description}</task>"

        if self._context:
            full_instruction += "<context>"
            for key, value in self._context.items():
                full_instruction += f"{key} - {value}"
            full_instruction += "</context>"

        if (self._memories is not None) and (len(self._memories)) > 0:

            # full_instruction += f" Relevant information: {self._memories}"
            for item in self._memories:
                full_instruction += f"\n<memory> {item} </memory>"

        return full_instruction

    def __repr__(self) -> str:
        return (
            f"Instruction(description={self._description!r}, "
            f"context={self._context!r}, memories={self._memories!r})"
        )

    def __str__(self) -> str:
        return self.render()


class ReplanningInstruction:
    """
    Definition of the replanning step instruction which instead of a single task
    takes the current taskqueue (completed and pending tasks) and based on the progress
    (represented by the results of the completed tasks) performs the replanning - potential
    change of the pending tasks (done tasks are not changed).
    """

    def __init__(self, taskqueue: TaskQueue, original_instruction: str) -> None:
        """
        Initialize the replanning instruction with the taskqueue and progress.
        """

        self._taskqueue = taskqueue
        self._original_instruction = original_instruction

    def render(self) -> str:
        """
        Create full string representation of the replanning instruction including taskqueue
        and progress.
        """

        full_instruction = "<plan>"
        for _, task in self._taskqueue.get_all_tasks().items():

            full_instruction += (
                "<JSON>"
                f"'id': '{task.id}',\n"
                f"'description': '{task.description}',\n"
                f"'dependencies': '{task.dependencies}',\n"
                f"'result': '{task.result}',\n"
                f"'status': '{task.status}'"
                "</JSON>"
            )

        full_instruction += "</plan>\n"

        full_instruction += f"Original instruction: {self._original_instruction}"

        return full_instruction

    def __str__(self) -> str:
        return self.render()


class ComposerInstruction:
    """
    Definition of the composer instruction which instead of a single task
    takes the current taskqueue (completed and pending tasks) and based on the progress
    (represented by the results of the completed tasks) performs the composition of the final
    response.
    """

    def __init__(self, taskqueue: TaskQueue, original_instruction: str) -> None:
        """
        Initialize the composer instruction with the taskqueue and progress.
        """

        self._taskqueue = taskqueue
        self._original_instruction = original_instruction

    def render(self) -> str:
        """
        Create full string representation of the composer instruction including taskqueue
        and progress.
        """

        full_instruction = (
            f"<instruction> {self._original_instruction} </instruction>\n"
        )
        for _, task in self._taskqueue.get_all_tasks().items():

            full_instruction += (
                "<partial_results>"
                f"'subtask': '{task.description}',\n"
                f"'result': '{task.result}',\n"
                "</partial_results>"
            )

        return full_instruction

    def __str__(self) -> str:
        return self.render()
