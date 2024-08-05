"""
Unit testing utilities for `gempyor`

This module contains unit testing utilities, mostly pytest fixtures. To use this module
the optional test dependencies must be installed.
"""

__all__ = [
    "change_directory_to_temp_directory",
    "create_confuse_rootview_from_dict",
    "create_confuse_subview_from_dict",
    "partials_are_similar",
]

from collections.abc import Generator
import functools
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


def partials_are_similar(
    f: functools.partial,
    g: functools.partial,
    check_func: bool = True,
    check_args: bool = True,
    check_keywords: bool = True,
) -> bool:
    """
    Check if two partials are 'similar' enough to be equal.

    For most unit testing purposes python's default `__eq__` method does not have the
    desired behavior for `functools.partial`. For unit testing purposes it is usually
    sufficient that two partials are similar enough. See python/cpython#47814 for more
    details on why `__eq__` is tricky for `functools.partial`.

    Args:
        f: A partial function to test.
        g: A partial function to test.
        check_func: If the `func` attributes of `f` and `g` should be checked for
            equality.
        check_args: If the `args` attributes of `f` and `g` should be checked for
            equality.
        check_keywords: If the `keywords` attributes of `f` and `g` should be checked
            for equality.

    Returns:
        A boolean indicating if `f` and `g` are similar.

    Examples:
        >>> from functools import partial
        >>> a = lambda x, y: x + y
        >>> b = partial(a, 1)
        >>> c = partial(a, 1.)
        >>> b == c
        False
        >>> partials_are_similar(b, c)
        True
    """
    if check_func and f.func != g.func:
        return False
    elif check_args and f.args != g.args:
        return False
    elif check_keywords and f.keywords != g.keywords:
        return False
    return True
