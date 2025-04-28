"""
Decorator for Singleton pattern.
"""

from typing import Any, Callable, Type, TypeVar

T = TypeVar("T")


def singleton(cls: Type[T]) -> Callable[..., T]:
    """
    Singleton pattern decorator.

    :param cls: class to be decorated.
    :return: decorated class.
    """

    instances: dict[Type[T], T] = {}

    def getinstance(*args: Any, **kwargs: Any) -> T:
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return getinstance
