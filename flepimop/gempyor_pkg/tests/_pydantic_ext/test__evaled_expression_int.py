"""Tests for the `gempyor._pydantic_ext._evaled_expression_int` function."""

import pytest

from gempyor._pydantic_ext import _evaled_expression_int


@pytest.mark.parametrize(
    ("val", "expected"),
    [
        ("2 * 5", 10),
        ("sqrt(81)", 9),
        ("11 / 2", 5),
        ("3.99", 3),
        (99, 99),
        (-50, -50),
        (None, None),
        (True, True),
        (10.5, 10.5),
        ([1, 2], [1, 2]),
    ],
)
def test_evaled_expression_int_returns_expected_results(val, expected):
    assert _evaled_expression_int(val) == expected


@pytest.mark.parametrize(
    "val",
    [
        "a * b",
        "not_a_valid_expression",
        "1 / 0",
    ],
)
def test_evaled_expression_int_raises_value_error_for_invalid_input(val):
    with pytest.raises(ValueError):
        _evaled_expression_int(val)
