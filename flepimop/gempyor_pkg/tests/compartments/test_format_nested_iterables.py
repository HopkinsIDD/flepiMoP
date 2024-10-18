from typing import Any, Iterable

import pandas as pd
import pytest

from gempyor.compartments import _format_nested_iterables


@pytest.mark.parametrize(
    "x", ([[]], [[], ["1", "2"]], [["x", "y", "z"], [], ["1", "2", "3"]])
)
def test_cannot_accept_a_list_of_empty_list(x: Iterable[Iterable[str]]) -> None:
    with pytest.raises(
        TypeError, match=r"^reduce\(\) of empty iterable with no initial value$"
    ):
        _format_nested_iterables(x)


@pytest.mark.parametrize(
    ("x", "expected"),
    (
        ([], []),
        ((), []),
        (set(), []),
        (pd.Series(), []),
        ([["a"], ["b"], ["c"]], ["a", "b", "c"]),
        (({"a"}, ("b"), ["c"]), ["a", "b", "c"]),
        ((("a", "a", "a"), ["x", "y", "z"]), ["a_a_a", "x_y_z"]),
        (
            pd.Series(data=[["a", "b", "c"], ["m"], ["x", "y", "z"]]),
            ["a_b_c", "m", "x_y_z"],
        ),
    ),
)
def test_format_nested_iterables_output_validation(
    x: Iterable[Iterable[str]], expected: list[str]
) -> None:
    actual = _format_nested_iterables(x)
    assert actual == expected
