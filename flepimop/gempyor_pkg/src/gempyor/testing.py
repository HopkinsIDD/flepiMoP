"""
Unit testing utilities for `gempyor`

This module contains unit testing utilities, mostly pytest fixtures. To use this module
the optional test dependencies must be installed.
"""

__all__ = [
    "change_directory_to_temp_directory",
    "create_confuse_configview_from_dict",
    "partials_are_similar",
    "sample_fits_distribution",
]

from collections.abc import Generator
import functools
import os
from tempfile import TemporaryDirectory
from typing import Any, Literal

import confuse
import numpy as np
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


def create_confuse_configview_from_dict(
    data: dict[str, Any], name: None | str = None
) -> confuse.ConfigView:
    """
    Create a ConfigView from a dictionary for unit testing confuse parameters.

    Args:
        data: The data to populate the confuse ConfigView with.
        name: The name of the Subview being created or if is `None` a RootView is
            created instead.

    Returns:
        Either a confuse Subview or RootView depending on the value of `name`.

    Examples:
        >>> data = {
        ...     "foo": "bar",
        ...     "fizz": 123,
        ...     "alphabet": ["a", "b", "c"],
        ...     "mapping": {"x": 1, "y": 2},
        ... }
        >>> rv = create_confuse_configview_from_dict(data)
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
        >>> sv = create_confuse_configview_from_dict(data, "params")
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
    data = {name: data} if name is not None else data
    cv = confuse.RootView([confuse.ConfigSource.of(data)])
    cv = cv[name] if name is not None else cv
    return cv


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


def sample_fits_distribution(
    sample: float | int,
    distribution: Literal[
        "fixed", "uniform", "poisson", "binomial", "truncnorm", "lognorm"
    ],
    **kwargs: dict[str, Any],
) -> bool:
    """
    Test if a sample fits a distribution with a given set of parameters.

    This function tests if the given `sample` could possibly be drawn from the
    distribution given with its parameters, but it does not test if it could reasonably
    be drawn from that distribution.

    Args:
        sample: The value to test.
        distribution: The name of the distribution to test against.
        **kwargs: Further arguments to specify the parameters of a distribution.

    Returns:
        A boolean indicating if the sample given could be from the distribution.

    See Also:
        gempyor.utils.random_distribution_sampler

    Examples:
        >>> sample_fits_distribution(0.0, "fixed", value=0.0)
        True
        >>> sample_fits_distribution(0.0, "fixed", value=0.5)
        False
        >>> sample_fits_distribution(0.5, "poisson", lam=3.0)
        False
        >>> sample_fits_distribution(
        ...     -3.5, "truncnorm", a=-5.5, b=3.4, mean=-1.4, sd=1.1
        ... )
        True
        >>> sample_fits_distribution(100000000, "lognorm", meanlog=1.0, sdlog=1.0)
        True
    """
    # Poisson and binomial only have support on a subset of the integers
    if distribution in ["poisson", "binomial"] and not (
        isinstance(sample, int) or (isinstance(sample, float) and sample.is_integer())
    ):
        return False
    # Now check distribution constraints
    if distribution == "fixed":
        return bool(np.isclose(sample, kwargs.get("value")))
    elif distribution == "uniform":
        # Uniform is on [low,high), but want uniform to match fixed when low == high.
        return bool(
            (
                np.isclose(kwargs.get("high"), kwargs.get("low"))
                and np.isclose(sample, kwargs.get("low"))
            )
            or (
                np.greater_equal(sample, kwargs.get("low"))
                and np.less(sample, kwargs.get("high"))
            )
        )
    elif distribution == "poisson":
        return bool(np.greater_equal(sample, 0.0))
    elif distribution == "binomial":
        return bool(
            np.greater_equal(sample, 0.0) and np.less_equal(sample, kwargs.get("n"))
        )
    elif distribution == "truncnorm":
        return bool(
            np.greater(sample, kwargs.get("a")) and np.less(sample, kwargs.get("b"))
        )
    elif distribution == "lognorm":
        return bool(np.greater(sample, 0.0))
