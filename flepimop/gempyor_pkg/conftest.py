"""Pytest configuration file."""

from collections.abc import Generator

import pytest


@pytest.fixture(autouse=True)
def _docdir(request: pytest.FixtureRequest) -> Generator[None, None, None]:
    # Run doctests in the temporary directory.
    # https://stackoverflow.com/q/46962007
    doctest_plugin = request.config.pluginmanager.getplugin("doctest")
    if isinstance(request.node, doctest_plugin.DoctestItem):
        tmpdir = request.getfixturevalue("tmpdir")
        with tmpdir.as_cwd():
            yield
    else:
        yield
