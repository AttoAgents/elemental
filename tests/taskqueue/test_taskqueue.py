"""
Test task queue operations.
"""

import json
import unittest
from typing import Any, Dict, List

from elemental_agents.core.taskqueue.queue import TaskQueue
from elemental_agents.core.taskqueue.task import Status
from elemental_agents.utils.utils import extract_tag_content


class TestTaskQueue(unittest.TestCase):
    """
    Test task queue operations.
    """

    def test_task_queue(self) -> None:
        """
        Test creating a task queue.
        """

        task_queue = TaskQueue()

        original_instruction = "Test instruction"

        tasks: List[Dict[str, Any]] = [
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

        task_queue.create_task_queue(tasks, original_instruction)

        self.assertEqual(len(task_queue.get_all_tasks()), len(tasks))

    def test_get_next_ready_tasks(self) -> None:
        """
        Test getting the next ready task.
        """

        task_queue = TaskQueue()

        original_instruction = "Test instruction"

        tasks: List[Dict[str, Any]] = [
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

        task_queue.create_task_queue(tasks, original_instruction)

        (next_task, queue_status) = task_queue.get_next_ready_tasks()

        self.assertEqual(next_task.description, "Task 1")
        self.assertEqual(queue_status, Status.IN_PROGRESS)

    def test_get_next_ready_consecutive(self) -> None:
        """
        Test consecutive calls to get the next ready task.
        """

        task_queue = TaskQueue()

        original_instruction = "Test instruction"

        tasks: List[Dict[str, Any]] = [
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

        task_queue.create_task_queue(tasks, original_instruction)
        (next_task, queue_status) = task_queue.get_next_ready_tasks()
        self.assertEqual(next_task.description, "Task 1")
        self.assertEqual(queue_status, Status.IN_PROGRESS)

        (next_task, queue_status) = task_queue.get_next_ready_tasks()
        self.assertEqual(next_task, None)

    def test_planner_generated_plan(self) -> None:
        """
        Test task queue with a planner-generated plan.
        """

        agent_raw_response = """
<thoughts>Considering the question about the color of the sky and thinking about what contributes to this phenomenon. The sky appears blue during the day due to a combination of factors, including the scattering of sunlight by the atmosphere and the way our eyes perceive light. This is a complex topic that can be broken down into simpler questions.

To answer why the sky is blue, we need to consider the following:

* What happens when sunlight enters Earth's atmosphere?
* How does this affect the wavelengths of light that reach our eyes?
* Why do shorter (blue) wavelengths dominate in the visible spectrum?

These are all related but distinct concepts that can be addressed step-by-step.</thoughts>
<plan>
<JSON>
{
"id": 1,
"description": "Determine the path of sunlight as it enters Earth's atmosphere",
"dependencies": []
}
</JSON>
<JSON>
{
"id": 2,
"description": "Explain how atmospheric particles scatter sunlight, affecting its wavelength",
"dependencies": [1]
}
</JSON>
<JSON>
{
"id": 3,
"description": "Describe why shorter wavelengths (like blue light) are more prominent in the visible spectrum",
"dependencies": [2]
}
</JSON>
<JSON>
{
"id": 4,
"description": "Combine the information from steps 1-3 to explain why the sky appears blue during the day",
"dependencies": [1, 2, 3]
}
</JSON>
</plan>
        """

        plan = extract_tag_content(agent_raw_response, "JSON")
        parsed_plan = [json.loads(p) for p in plan]

        task_queue = TaskQueue()
        original_instruction = "Test instruction"
        task_queue.create_task_queue(parsed_plan, original_instruction)

        task_queue.print_tasks()

        (next_task, queue_status) = task_queue.get_next_ready_tasks()

        self.assertIsNotNone(next_task.description)
        self.assertEqual(queue_status, Status.IN_PROGRESS)


if __name__ == "__main__":
    unittest.main()
