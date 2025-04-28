"""
Generic factory function to dynamically instantiate a class from a string.
"""

import importlib
from typing import Any

from loguru import logger


def factory(full_class_string: str, *args: Any, **kwargs: Any) -> Any:
    """
    Dynamically instantiate a class from a string. Example usage:
    instance = factory('myclasses.ClassA', x=10)

    :param full_class_string: String of the form 'package.module.ClassName'
    :param args: Positional arguments for class instantiation
    :param kwargs: Keyword arguments for class instantiation
    :return: An instance of the specified class
    """

    module_path, class_name = full_class_string.rsplit(".", 1)
    try:
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        return cls(*args, **kwargs)
    except (ImportError, AttributeError) as e:
        logger.error(f"Cannot import '{class_name}' from '{module_path}'. Error: {e}")
        raise ImportError(f"Cannot import '{class_name}' from '{module_path}'") from e
