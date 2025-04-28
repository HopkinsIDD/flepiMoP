"""Unit tests for the `gempyor.sync._sync_filter._filter_pattern` function."""

from typing import Literal

import pytest

from gempyor.sync._sync_filter import _filter_pattern


@pytest.mark.parametrize("mode", ["- ", "+ ", ""])
@pytest.mark.parametrize(
    "path", ["/foo/bar.txt", "/fizz/buzz.txt", "/a/b/c.txt", "x/y/z.txt"]
)
def test_filter_pattern(mode: Literal["- ", "+ ", ""], path: str) -> None:
    """
    Test the `_filter_pattern` function with various filter strings.

    Args:
        mode: The mode prefix to test.
        path: The path to test.
    """
    assert _filter_pattern(mode + path) == path
