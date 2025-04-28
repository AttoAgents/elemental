"""
Collection of utility functions.
"""

import os
import random
import re
import shutil
import string
from pathlib import Path
from typing import List

from loguru import logger


def get_random_string(max_length: int = 10) -> str:
    """
    Generate a random string.

    :param max_length: Maximum length of the random string.
    :return: Random string.
    """

    my_id = "".join(
        random.choices(string.ascii_uppercase + string.digits, k=max_length)
    )

    return my_id


def remove_directory_if_exists(directory_path: str) -> None:
    """
    Remove a directory if it exists.

    :param directory_path: Path to the directory to remove.
    """
    if os.path.exists(directory_path):
        if os.path.isdir(directory_path):
            shutil.rmtree(directory_path)
            logger.debug(f"Directory '{directory_path}' removed.")
        else:
            logger.debug(f"'{directory_path}' exists but is not a directory.")
    else:
        logger.error(f"Directory '{directory_path}' does not exist.")


def extract_tag_content(text: str, tag: str) -> List[str]:
    """
    Extract content between XML-like tags in a text string.

    :param text: Text string to search for tags.
    :param tag: Tag to search for.
    :return: List of content between tags.
    """
    # Regular expression pattern to find content between <tag>...</tag>
    pattern = rf"<{tag}>(.*?)</{tag}>"

    # Find all matches for the pattern
    matches = re.findall(pattern, text, re.DOTALL)

    return matches


def extract_unclosed_tag_content(text: str, tag: str) -> List[str]:
    """
    Extract content between XML-like tags in a text string that are not closed.

    :param text: Text string to search for tags.
    :param tag: Tag to search for.
    :return: List of content between tags.
    """
    pattern = f"<{tag}>"

    # Find all matches for the pattern
    matches = text.split(pattern)[1:]

    return matches


def is_path_in_current_or_subdirectory(file_path: str) -> bool:
    """
    Check if a file path is in the current directory or a subdirectory.

    :param file_path: Path to the file to check.
    :return: True if the file path is in the current directory or a subdirectory, False otherwise.
    """
    # Convert the current directory and provided path to absolute paths
    current_directory = Path(os.getcwd()).resolve()
    provided_path = Path(file_path).resolve()

    try:
        # Check if the provided path starts with the current directory
        return provided_path.is_relative_to(current_directory)
    except AttributeError:
        # For Python versions < 3.9, use a different method
        return str(provided_path).startswith(str(current_directory))


def split_text_into_chunks(
    text: str, chunk_length: int, chunk_overlap: int
) -> List[str]:
    """
    Split a text into chunks of a specified length with an overlap.

    :param text: Text to split into chunks.
    :param chunk_length: Length of each chunk.
    :param chunk_overlap: Number of words to overlap between chunks.
    :return: List of text chunks.
    """

    # Split the text into words
    words = text.split()

    # Initialize variables
    chunks = []
    start = 0

    # Iterate through the words with a sliding window
    while start < len(words):
        # End index for the current chunk
        end = start + chunk_length
        # Extract the chunk
        chunk = words[start:end]
        # Join the words back into a string
        chunks.append(" ".join(chunk))
        # Move the start index forward by chunk_length - chunk_overlap
        start += chunk_length - chunk_overlap

    return chunks
