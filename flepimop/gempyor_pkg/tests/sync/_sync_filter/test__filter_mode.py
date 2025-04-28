"""Unit tests for the `gempyor.sync._sync_filter._filter_mode` function."""

from typing import Literal

import pytest

from gempyor.sync._sync_filter import _filter_mode


@pytest.mark.parametrize("mode", ["- ", "+ ", ""])
@pytest.mark.parametrize(
    "path", ["/foo/bar.txt", "/fizz/buzz.txt", "/a/b/c.txt", "x/y/z.txt"]
)
def test_filter_mode(mode: Literal["- ", "+ ", ""], path: str) -> None:
    """
    Test the `_filter_mode` function with various filter strings.

    Args:
        mode: The mode prefix to test.
        path: The path to test.
    """
    assert _filter_mode(mode + path) == mode.strip() or "+"
