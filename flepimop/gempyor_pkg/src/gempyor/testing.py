"""
Unit testing utilities for `gempyor`

This module contains unit testing utilities, mostly pytest fixtures. To use this module
the optional test dependencies must be installed.
"""

__all__ = [
    "change_directory_to_temp_directory",
    "create_confuse_rootview_from_dict",
    "create_confuse_subview_from_dict",
]

from collections.abc import Generator
import os
from tempfile import TemporaryDirectory
from typing import Any

import confuse
import pytest


@pytest.fixture
def change_directory_to_temp_directory() -> Generator[None, None, None]:
    """Change test working directory to a temporary directory

    Pytest fixture that will create a temporary directory and change the working
    directory to that temporary directory. This fixture also cleans up after itself by
    resetting the working directory and removing the temporary directory on test end.
    Useful for testing functions that create files relative to the working directory.

    Yields:
        None
    """
    current_dir = os.getcwd()
    temp_dir = TemporaryDirectory()
    os.chdir(temp_dir.name)
    yield
    os.chdir(current_dir)
    temp_dir.cleanup()


def create_confuse_rootview_from_dict(data: dict[str, Any]) -> confuse.RootView:
    """
    Create a RootView from a dictionary for unit testing confuse parameters.

    Args:
        data: The data to populate the confuse root view with.

    Returns:
        A confuse root view.

    Examples:
        >>> data = {
        ...     "foo": "bar",
        ...     "fizz": 123,
        ...     "alphabet": ["a", "b", "c"],
        ...     "mapping": {"x": 1, "y": 2},
        ... }
        >>> rv = create_confuse_rootview_from_dict(data)
        >>> rv
        <RootView: root>
        >>> rv.keys()
        ['foo', 'fizz', 'alphabet', 'mapping']
        >>> rv.get()
        {'foo': 'bar', 'fizz': 123, 'alphabet': ['a', 'b', 'c'], 'mapping': {'x': 1, 'y': 2}}
        >>> rv == rv.root()
        True
        >>> rv.name
        'root'
    """
    return confuse.RootView([confuse.ConfigSource.of(data)])


def create_confuse_subview_from_dict(
    name: str, data: dict[str, Any]
) -> confuse.Subview:
    """
    Create a Subview from a dictionary for unit testing confuse parameters.

    Args:
        name: The name of the subview being created.
        data: The data to populate the confuse subview with.

    Returns:
        A confuse subview.

    Examples:
        >>> data = {
        ...     "foo": "bar",
        ...     "fizz": 123,
        ...     "alphabet": ["a", "b", "c"],
        ...     "mapping": {"x": 1, "y": 2},
        ... }
        >>> sv = create_confuse_subview_from_dict("params", data)
        >>> sv
        <Subview: params>
        >>> sv.keys()
        ['foo', 'fizz', 'alphabet', 'mapping']
        >>> sv.get()
        {'foo': 'bar', 'fizz': 123, 'alphabet': ['a', 'b', 'c'], 'mapping': {'x': 1, 'y': 2}}
        >>> sv == sv.root()
        False
        >>> sv.name
        'params'
    """
    root_view = create_confuse_rootview_from_dict({name: data})
    return root_view[name]
