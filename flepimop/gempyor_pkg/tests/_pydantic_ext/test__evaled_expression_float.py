"""Tests for the `gempyor._pydantic_ext._evaled_expression_float` function."""

import pytest

from gempyor._pydantic_ext import _evaled_expression_float


@pytest.mark.parametrize(
    ("val", "expected"),
    [
        ("1 + 1", 2.0),
        ("5 / 2", 2.5),
        ("sqrt(9)", 3.0),
        (99.5, 99.5),
        (-10.0, -10.0),
        (None, None),
        (True, True),
        ([1, 2, 3], [1, 2, 3]),
        ({"a": 1}, {"a": 1}),
    ],
)
def test_evaled_expression_float_returns_expected_results(val, expected):
    assert _evaled_expression_float(val) == expected


@pytest.mark.parametrize(
    "val",
    [
        "a * b",
        "not_a_valid_expression",
        "1 / 0",
    ],
)
def test_evaled_expression_float_raises_value_error_for_invalid_input(val):
    with pytest.raises(ValueError):
        _evaled_expression_float(val)
