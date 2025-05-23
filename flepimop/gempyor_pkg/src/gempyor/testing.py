"""
Unit testing utilities for `gempyor`

This module contains unit testing utilities, mostly pytest fixtures. To use this module
the optional test dependencies must be installed.
"""

__all__ = [
    "change_directory_to_temp_directory",
    "create_confuse_config_from_file",
    "create_confuse_configview_from_dict",
    "create_confuse_config_from_dict",
    "mock_empty_config",
    "partials_are_similar",
    "sample_fits_distribution",
    "setup_example_from_tutorials",
    "run_test_in_separate_process",
]

from collections.abc import Generator, Iterable
import functools
import os
from pathlib import Path
from shlex import quote
import shutil
from stat import S_IXUSR
import subprocess
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import Any, Literal

import confuse
import numpy as np
import pytest

from .utils import _shutil_which


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


def mock_empty_config() -> confuse.Configuration:
    """
    Create a `confuse.Configuration` (akin to `gempyor.utils.config`) with no data for
    unit testing configurations.

    Returns:
        A `confuse.Configuration`.
    """
    return confuse.Configuration("flepiMoPMock", read=False)


def create_confuse_config_from_file(
    data_file: Path,
) -> confuse.Configuration:
    """
    Create a `confuse.Configuration` (akin to `gempyor.utils.config`) from a file for
    unit testing configurations.

    Args:
        data_file: The file to populate the confuse ConfigView with.

    Returns:
        A `confuse.Configuration`.
    """
    cv = mock_empty_config()
    cv.set_file(data_file)
    return cv


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
        This example gives a brief demonstration of how to represent this yaml:
        ```yaml
        foo: bar
        fizz: 123
        alphabet: [a, b, c]
        mapping:
            x: 1
            y: 2
        ```
        with this function as a python dict for unit testing purposes.
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


def create_confuse_config_from_dict(data: dict[str, Any]) -> confuse.Configuration:
    """
    Create a Configuration from a dictionary for unit testing confuse parameters.

    Args:
        data: The data to populate the confuse ConfigView with.

    Returns:
        confuse Configuration


    """
    cfg = mock_empty_config()
    cfg.set_args(data)
    return cfg


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


def run_test_in_separate_process(
    script: str | Path, dest: Path | None = None, args: Iterable[str] = ()
) -> None:
    """
    Execute a test script in a separate process.

    This function is useful for testing functionality that requires initialization for
    the process (i.e. setting the start method for multiprocessing). This method of unit
    testing is **slow** and should be used sparingly.

    Args:
        script: The script to run, either a string of the code or a path to the script
            to use.
        dest: The destination to write the script to or `None` to use a temporary file.
        args: The arguments to pass to the script, can be obtained in the script either
            through `sys.argv` (starting with index 1) or through `argparse`.

    Returns:
        None

    Examples:
        Typically this function is used in a test like so:

        >>> assert run_test_in_separate_process(
        ...     Path(__file__).parent / "external_script.py",
        ...     tmp_path / "test.py",
        ...     args=["arg1", "arg2"],
        ... ) is None

        The `external_script.py` would reside in the same directory as the test file and
        the `tmp_path` would be a `pytest` fixture.
    """
    if not isinstance(script, Path):
        if dest is None:
            with NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(script.encode())
                script = Path(temp_file.name)
        else:
            dest.write_text(script)
            script = dest
    elif dest is not None:
        shutil.copy(script, dest)
        script = dest

    try:
        python = _shutil_which("python")
        args = [python, str(script)] + list(quote(a) for a in args)
        proc = subprocess.run(args, capture_output=True, check=True)
        return proc.returncode
    finally:
        if dest is None and script.exists():
            script.unlink()


def sample_script(directory: Path, executable: bool, name: str = "example") -> Path:
    """
    Create a sample script for testing functions that require a script.

    Args:
        directory: The directory to create the script in.
        executable: If the script should be executable.
        name: The name of the script.

    Returns:
        The path to the script.

    Notes:
        The script is a simple bash script in a file named `name` with contents:
        ```bash
        #!/usr/bin/env bash
        echo 'Hello local!'
        ```
    """
    script = directory / name
    script.write_text("#!/usr/bin/env bash\necho 'Hello local!'")
    if executable:
        script.chmod(script.stat().st_mode | S_IXUSR)
    return script


def setup_example_from_tutorials(
    tmp_path: Path,
    config: str,
) -> None:
    """
    Setup a tutorial example for testing.

    Args:
        tmp_path: The temporary directory to create the example in.
        config: The name of the configuration file to use.

    Returns:
        The path to the temporary directory.

    Raises:
        ValueError: If the `FLEPI_PATH` environment variable is not set.
    """
    if (flepi_path := os.getenv("FLEPI_PATH")) is None:
        raise ValueError("FLEPI_PATH environment variable is not set.")
    tutorials_path = Path(flepi_path) / "examples/tutorials"
    for file in [config] + [
        f.relative_to(tutorials_path)
        for f in tutorials_path.glob("model_input/*")
        if f.is_file()
    ]:
        source = tutorials_path / file
        destination = tmp_path / file
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(source, destination)
