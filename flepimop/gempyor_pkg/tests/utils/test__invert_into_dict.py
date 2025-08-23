"""Unit tests for the `_invert_into_dict` internal utility"""

from typing import TypeVar

import pytest

from gempyor.utils import _invert_into_dict


T = TypeVar("T")
U = TypeVar("U")


@pytest.mark.parametrize(
    ("value_one", "keys_one", "expected_one", "value_two", "keys_two", "expected_two"),
    [
        (1, [1, 2], {1: [1], 2: [1]}, 2, [2, 3], {1: [1], 2: [1, 2], 3: [2]}),
        (
            "a",
            ["x", "y"],
            {"x": ["a"], "y": ["a"]},
            "a",
            ["y", "z"],
            {"x": ["a"], "y": 2 * ["a"], "z": ["a"]},
        ),
        (
            1.2,
            ["abc", "def", "ghi"],
            {"abc": [1.2], "def": [1.2], "ghi": [1.2]},
            3.4,
            ["abc", "def", "ghi", "jkl"],
            {"abc": [1.2, 3.4], "def": [1.2, 3.4], "ghi": [1.2, 3.4], "jkl": [3.4]},
        ),
        (
            "foobar",
            ["abc", "def"],
            {"abc": ["foobar"], "def": ["foobar"]},
            "fizzbuzz",
            [],
            {"abc": ["foobar"], "def": ["foobar"]},
        ),
        (
            True,
            [],
            {},
            False,
            ["yes", "no"],
            {"yes": [False], "no": [False]},
        ),
        (
            123,
            [],
            {},
            456,
            [],
            {},
        ),
    ],
)
def test_exact_results_for_select_inputs(
    value_one: U,
    keys_one: list[T],
    expected_one: dict[T, list[U]],
    value_two: U,
    keys_two: list[T],
    expected_two: dict[T, list[U]],
) -> None:
    """Check that successive calls produce exact expected results."""
    resulting_dict = {}
    assert _invert_into_dict(resulting_dict, value_one, keys_one) is None
    assert resulting_dict == expected_one
    assert _invert_into_dict(resulting_dict, value_two, keys_two) is None
    assert resulting_dict == expected_two
