"""
Taskqueue definition.
"""

from typing import Any, Dict, List, Tuple

from loguru import logger
from pydantic import BaseModel
from rich.console import Console

from elemental_agents.core.taskqueue.task import Status, Task
from elemental_agents.utils.utils import get_random_string


class TaskQueue(BaseModel):
    """
    TaskQueue class to represent a task queue. A task queue has a list of tasks
    and a status. The status of a task queue can be READY, BLOCKED, DONE, FAILED
    or IN_PROGRESS. The task queue can create a task queue from a list of tasks,
    update the status of a task, get a task by ID, remove a task, get the status
    of the task queue, and get the next ready task.
    """

    _tasks: Dict[str, Task] = {}
    _status: Status = Status.BLOCKED

    def create_task_queue(
        self,
        tasks: List[Dict[str, Any]],
        original_instruction: str,
        keep_ids: bool = False,
    ) -> Status:
        """
        Create a task queue with a list of tasks. The task queue is initialized
        from the list of strings that have JSON representation of tasks. This
        creates Task objects from the JSON strings and adds them to the task
        queue.

        :param tasks: List of tasks
        :param original_instruction: Original instruction from which the task
            queue is created
        :param keep_ids: If True, the IDs of the tasks are kept as they are.
            If False, the IDs are replaced with standardized random strings.
        :return: Status
        """
        self._tasks = {}
        self._status = Status.BLOCKED

        # Create tasks from the list of strings and add them to the task queue.
        task_list = []
        for item in tasks:

            # Convert int to strings in id and dependencies
            item["id"] = str(item["id"])
            item["dependencies"] = [str(dep) for dep in item["dependencies"]]
            item["origin"] = original_instruction

            task_obj = Task(**item)
            task_list.append(task_obj)

        logger.info(f"Task list: {task_list}")

        # Revise the IDs and assign a unique and standardized ID to the task
        for task in task_list:

            # Change ids unless keep_ids is True
            if not keep_ids:
                old_id = task.id
                new_id = get_random_string()
                task.id = new_id

                # Update the dependencies to refer to the new ID
                for other_task in task_list:
                    other_task.dependencies = [
                        new_id if dep == old_id else dep
                        for dep in other_task.dependencies
                    ]

            # Set the status to READY if the task has no dependencies
            if len(task.dependencies) == 0:
                task.status = Status.READY
            else:
                task.status = Status.BLOCKED

        # Add the tasks to the task queue
        for task in task_list:
            my_id = task.id
            self._tasks[my_id] = task

        # Set the status of the task queue to READY if the task queue is created
        self._status = Status.READY
        return self._status

    def mark_tasks_with_results_as_done(self) -> None:
        """
        Mark all tasks with results as DONE.
        """
        for _, task in self._tasks.items():
            if task.result != "":
                task.status = Status.DONE

        self.update_tasks_statuses()

    def update_tasks_statuses(self) -> None:
        """
        Update the status of the tasks in the task queue. If a task has all its
        dependencies done, set the status of the task to READY.
        """
        for _, task in self._tasks.items():
            if task.status == Status.BLOCKED:
                if all(
                    self._tasks[dep].status == Status.DONE for dep in task.dependencies
                ):
                    task.status = Status.READY

    def add_context_to_all_tasks(self, identifier: str, information: str) -> None:
        """
        Add information to the context of all tasks in the task queue.

        :param identifier: The ID of the task whose result is being added or
            identifier of the context message.
        :param information: The information to be added to the context.
        """
        for key, _ in self._tasks.items():
            self._tasks[key].context[identifier] = information

    def fix_context_for_all_tasks(self) -> None:
        """
        Fix the context of all tasks in the task queue. This method assumes that
        the context has not been properly set and results of dependent tasks are
        not available.
        """

        for _, task in self._tasks.items():

            task_id = task.id
            if task.status == Status.DONE:

                # Add the result of the task to the context
                # of all tasks that depend on this task
                description = task.description
                for _, other_task in self._tasks.items():
                    if task_id in other_task.dependencies:
                        if description not in other_task.context:
                            other_task.context[description] = task.result

    def update_task(
        self, task_id: str, status: Status, result: str | None = None
    ) -> None:
        """
        Update the status of a task in the task queue.

        :param task_id: Task ID
        :param status: Status
        :param result: Result of the task
        """
        self._tasks[task_id].status = status
        self._tasks[task_id].result = result
        description = self._tasks[task_id].description

        # Add the result to the context of the all tasks that depend on this task
        for key, task in self._tasks.items():
            if task_id in task.dependencies:
                self._tasks[key].context[description] = result
                # self._tasks[key].context[task_id] = result

        # Set the status of the task queue to DONE if all tasks are done
        if all(task.status == Status.DONE for key, task in self._tasks.items()):
            self._status = Status.DONE

        # Update the status of the tasks in the task queue
        self.update_tasks_statuses()

    def get_task(self, task_id: str) -> Task:
        """
        Get a task from the task queue by task ID.

        :param task_id: Task ID
        :return: Task
        """
        return self._tasks[task_id]

    def get_all_tasks(self) -> Dict[str, Task]:
        """
        Get all tasks from the task queue.

        :return: Dict[str, Task]
        """
        return self._tasks

    def remove_task(self, task_id: str) -> None:
        """
        Remove a task from the task queue.

        :param task_id: Task ID
        """
        del self._tasks[task_id]

    def status(self) -> Status:
        """
        Returns the status of the task queue. Status can be READY, BLOCKED,
        DONE, FAILED or IN_PROGRESS.

        :return: Status
        """
        return self._status

    def get_next_ready_tasks(self) -> Tuple[Task, Status]:
        """
        Returns the next task that is ready to be executed. If no task is ready,
        returns the status of the task queue. Status can be DONE, FAILED if the
        whole task queue is done or failed respectively.

        :return: Tuple[Task, Status]
        """

        # Check the status of the task queue. If DONE or FAILED, return the
        # status. In any other case, proceed to get the next ready task.
        if self._status == Status.DONE:
            return (None, Status.DONE)
        if self._status == Status.FAILED:
            return (None, Status.FAILED)

        # Get next ready task from the task queue considering dependencies and
        # status of the task. This will be next available task in the READY
        # state.

        ready_tasks = []
        for _, task in self._tasks.items():
            if task.status == Status.READY:
                if all(
                    self._tasks[dep].status == Status.DONE for dep in task.dependencies
                ):
                    ready_tasks.append(task)

        if len(ready_tasks) > 0:
            # Set the status of the task queue to IN_PROGRESS if there are tasks
            # ready to be executed.
            self._status = Status.IN_PROGRESS

            # release the first task in the list and set its status to IN_PROGRESS
            ready_tasks[0].status = Status.IN_PROGRESS
            t_id = ready_tasks[0].id
            self._tasks[t_id].status = Status.IN_PROGRESS
            return (ready_tasks[0], Status.IN_PROGRESS)

        # If no task is ready, set the status of the task queue to BLOCKED
        self._status = Status.BLOCKED
        return (None, Status.BLOCKED)

    def print_tasks(self) -> None:
        """
        Print the tasks in the task queue.
        """

        console = Console()

        console.print(self._tasks)

    def revise_task_queue(
        self, revised_plan: List[Dict[str, Any]], original_instruction: str
    ) -> None:
        """
        Revise the task queue based on the revised plan. The revised plan
        follows the same structure as the original plan.

        :param revised_plan: Revised plan dictionary with the same structure as
            the original plan.
        :param original_instruction: Original instruction from which the task queue is created
        """

        # Create list of new tasks
        task_list = []
        for item in revised_plan:

            item["id"] = str(item["id"])
            item["dependencies"] = [str(dep) for dep in item["dependencies"]]
            item["origin"] = original_instruction

            task_obj = Task(**item)
            task_list.append(task_obj)

        # Check if the task is already in the task queue
        id_replacements = {}

        for new_task in task_list:
            new_task_id = new_task.id
            if new_task_id not in self._tasks:

                # New task

                # Revise the IDs and assign a unique and standardized ID to the task
                new_task.id = get_random_string()

                logger.info(f"Creating new task with ID: {new_task.id}")
                self._tasks[new_task.id] = new_task

                # Keep track of the ID replacements for dependencies
                id_replacements[new_task_id] = new_task.id

            else:
                # Update the task description
                self._tasks[new_task_id].description = new_task.description

                # Check if the dependencies are matching
                if self._tasks[new_task_id].dependencies != new_task.dependencies:
                    self._tasks[new_task_id].dependencies = new_task.dependencies

        # Update the dependencies to refer to the new ID
        for other_task in self._tasks.values():
            other_task.dependencies = [
                id_replacements[dep] if dep in id_replacements else dep
                for dep in other_task.dependencies
            ]

        # Check if any task should be removed
        new_task_ids = [task.id for task in task_list]
        to_remove = []
        for task_id, task in self._tasks.items():
            if (task_id not in new_task_ids) and (task.status != Status.DONE):
                to_remove.append(task_id)
                logger.info(f"Task {task_id} should be removed")

        for task_id in to_remove:
            self.remove_task(task_id)

        # Update the status of the task queue
        self.update_tasks_statuses()


if __name__ == "__main__":

    INSTRUCTION = "Test instruction"
    task_queue = TaskQueue()

    logger.info("Creating a task queue with two tasks")

    planned_tasks: list[Dict[str, Any]] = [
        {
            "id": "1",
            "description": "Task 1",
            "result": "",
            "status": Status.READY,
            "dependencies": [],
            "context": {},
        },
        {
            "id": "2",
            "description": "Task 2",
            "result": "",
            "status": Status.BLOCKED,
            "dependencies": ["1"],
            "context": {},
        },
        {
            "id": "3",
            "description": "Task 3",
            "result": "",
            "dependencies": ["1", "2"],
            "context": {},
        },
    ]

    task_queue.create_task_queue(planned_tasks, INSTRUCTION)
    task_queue.print_tasks()

    logger.info("Updating the status of the first task to DONE")
    t, s = task_queue.get_next_ready_tasks()
    current_id = t.id
    logger.info(f"Next ready task: {t}")
    task_queue.update_task(current_id, Status.DONE, "Task 1 result")
    task_queue.print_tasks()
