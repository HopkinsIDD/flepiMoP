"""Tests for the `gempyor._pydantic_ext._override_or_val` function."""


import pytest

from gempyor._pydantic_ext import _override_or_val


@pytest.mark.parametrize(
    ("override", "value", "expected"), 
    ((None, 1, 1), (None, "abc", "abc"), (1, "abc", 1), ("", "foo", "")),
)
def test_exact_results_for_select_values(override, value, expected):
    """Ensure that `_override_or_val` returns the expected results for specific inputs."""
    assert _override_or_val(override, value) == expected
