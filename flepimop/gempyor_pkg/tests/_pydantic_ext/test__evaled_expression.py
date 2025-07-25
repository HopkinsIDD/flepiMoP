"""Tests for the gempyor._pydantic_ext._evaled_expression function."""

import pytest

from gempyor._pydantic_ext import _evaled_expression


@pytest.mark.parametrize(
    ("val", "target_type", "expected"),
    [
        ("1 + 1", float, 2.0),
        ("5 / 2", float, 2.5),
        ("sqrt(9)", float, 3.0),
        (99.5, float, 99.5),
        (-10.0, float, -10.0),
        ("2 * 5", int, 10),
        ("sqrt(81)", int, 9),
        ("11 / 2", int, 5), 
        ("3.99", int, 3),    
        (99, int, 99),
        (-50, int, -50),
        (None, float, None),
        (None, int, None),
        (True, float, True),
        (True, int, True),
        (10.5, int, 10.5), 
        ([1, 2, 3], float, [1, 2, 3]),
        ({"a": 1}, float, {"a": 1}),
    ],
)
def test_evaled_expression_returns_expected_results(val, target_type, expected):
    assert _evaled_expression(val, target_type=target_type) == expected


@pytest.mark.parametrize(
    ("val", "target_type"),
    [
        ("a * b", int),
        ("a * b", float),
        ("not_a_valid_expression", int),
        ("not_a_valid_expression", float),
        ("1 / 0", int),
        ("1 / 0", float),
    ],
)
def test_evaled_expression_raises_value_error_for_invalid_input(val, target_type):
    with pytest.raises(ValueError):
        _evaled_expression(val, target_type=target_type)