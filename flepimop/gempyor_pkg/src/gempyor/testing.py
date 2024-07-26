"""
Unit testing utilities for `gempyor`

This module contains unit testing utilities, mostly pytest fixtures. To use this module
the optional test dependencies must be installed.
"""

from collections.abc import Generator
import os
from tempfile import TemporaryDirectory

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
