"""
Test the ListFiles tool.
"""

import json
import unittest

from elemental_agents.tools.list_files import ListFiles, ListFilesParams


class TestToolListFiles(unittest.TestCase):
    """
    Test listing files in the working directory.
    """

    def test_list_files(self) -> None:
        """
        Test listing only files in the working directory.
        """

        list_files = ListFiles()

        list_files_json = '{"path": ".", "files_only": true}'
        params = json.loads(list_files_json)
        result = list_files.run(ListFilesParams(**params))

        self.assertIsNotNone(str(result))
        self.assertTrue(len(result.files) > 0)
        self.assertFalse(result.is_in_list("elemental_agents"))
        self.assertTrue(result.is_in_list("README.md"))

    def test_list_files_and_directories(self) -> None:
        """
        Test listing files and directories in the working directory.
        """

        list_files = ListFiles()

        list_files_json = '{"path": ".", "files_only": false}'
        params = json.loads(list_files_json)
        result = list_files.run(ListFilesParams(**params))

        self.assertIsNotNone(str(result))
        self.assertTrue(len(result.files) > 0)
        self.assertTrue(result.is_in_list("elemental_agents"))
        self.assertTrue(result.is_in_list("README.md"))


if __name__ == "__main__":
    unittest.main()
