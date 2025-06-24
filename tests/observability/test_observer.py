"""
Test observer module for logging messages and tasks.
"""

import unittest

from elemental_agents.llm.data_model import Message
from elemental_agents.observability.observer import observer


class TestObserver(unittest.TestCase):
    """
    Test observer module.
    """

    def test_observer(self) -> None:
        """
        Basic test for observer.
        """

        (destination_type, destination_location) = observer.get_destination()

        self.assertIn(destination_type, ["screen", "file", "db", "none"])
        if destination_type == "file":
            self.assertIsNotNone(destination_location)
        if destination_type == "db":
            self.assertIsNotNone(destination_location)

        self.assertIsNotNone(observer)

    def test_singleton(self) -> None:
        """
        Test singelton pattern.
        """

        observer1 = observer
        observer2 = observer

        self.assertEqual(observer1, observer2)
        self.assertEqual(observer1._id, observer2._id)

    def test_log(self) -> None:
        """
        Test log message method.
        """

        msg = Message(role="system", content="Test system message")
        agent_name = "test_agent"
        task_description = "test_task"

        observer.log_message(
            input_session="TestSession",
            message=msg,
            agent_name=agent_name,
            task_description=task_description,
        )

        destination_type, _ = observer.get_destination()

        self.assertIn(destination_type, ["screen", "file", "db", "none"])


if __name__ == "__main__":
    unittest.main()
