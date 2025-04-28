"""
Test reading files tool.
"""

import json
import unittest

from elemental_agents.tools.read_files import ReadFiles, ReadFilesParams


class TestToolReadFiles(unittest.TestCase):
    """
    Test reading files into a dictionary.
    """

    def test_read_file(self) -> None:
        """
        Test reading a single file.
        """

        read_files = ReadFiles()

        read_files_json = '{"paths": ["./README.md"]}'
        params = json.loads(read_files_json)
        result = read_files.run(ReadFilesParams(**params))

        self.assertIsNotNone(str(result))

    def test_read_files(self) -> None:
        """
        Test reading multiple files.
        """

        read_files = ReadFiles()

        read_files_json = '{"paths": ["./README.md", "pyproject.toml"]}'
        params = json.loads(read_files_json)
        result = read_files.run(ReadFilesParams(**params))

        self.assertIn("README.md", str(result))
        self.assertIn("pyproject.toml", str(result))

    def test_read_files_error(self) -> None:
        """
        Test reading multiple files and non-existing file.
        """

        read_files = ReadFiles()

        read_files_json = (
            '{"paths": ["./README.md", "pyproject.toml", "nonexistentfile"]}'
        )
        params = json.loads(read_files_json)
        result = read_files.run(ReadFilesParams(**params))

        self.assertIn("README.md", str(result))
        self.assertIn("pyproject.toml", str(result))
        self.assertNotIn("nonexistentfile", str(result))

    def test_read_files_above_cwd(self) -> None:
        """
        Test reading files above the current working directory.
        """

        read_files = ReadFiles()

        read_files_json = (
            '{"paths": ["/etc/hosts", "pyproject.toml", "nonexistentfile"]}'
        )
        params = json.loads(read_files_json)
        result = read_files.run(ReadFilesParams(**params))

        self.assertNotIn("/etc/hosts", str(result))
        self.assertNotIn("nonexistentfile", str(result))
        self.assertIn("pyproject.toml", str(result))


if __name__ == "__main__":
    unittest.main()
