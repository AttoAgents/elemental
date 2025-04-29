"""
Test the write_files tool.
"""

import json
import os
import unittest

from elemental_agents.tools.write_file import WriteFile, WriteFileParams


class TestToolWriteFile(unittest.TestCase):
    """
    Test tool for writing files.
    """

    def test_write_file(self) -> None:
        """
        Write a single file.
        """

        write_files = WriteFile()

        write_files_json = (
            '{"file_path": "./test_write_files.txt", "content": ["test content"]}'
        )
        params = json.loads(write_files_json)
        result = write_files.run(WriteFileParams(**params))

        os.remove("./test_write_files.txt")

        self.assertIn("./test_write_files.txt", str(result))

    def test_write_files_error(self) -> None:
        """
        Test writing multiple files and a file above working directory.
        """

        write_files = WriteFile()

        write_files_json = '{"file_path": "/etc/hosts", "content": ["test content 1", "test content 2", "test content 3"]}'
        params = json.loads(write_files_json)
        result = write_files.run(WriteFileParams(**params))

        self.assertIn("ERROR", str(result))


if __name__ == "__main__":
    unittest.main()
