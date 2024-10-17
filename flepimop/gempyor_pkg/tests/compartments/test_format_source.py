from typing import Any, Iterable

import pandas as pd
import pytest

from gempyor.compartments import _format_source


@pytest.mark.parametrize(
    "source_column", ([[]], [[], ["1", "2"]], [["x", "y", "z"], [], ["1", "2", "3"]])
)
def test_cannot_accept_a_list_of_empty_list_source_column(
    source_column: Iterable[Iterable[Any]],
) -> None:
    with pytest.raises(
        TypeError, match=r"^reduce\(\) of empty iterable with no initial value$"
    ):
        _format_source(source_column)


@pytest.mark.parametrize(
    ("source_column", "expected"),
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
def test_format_source_output_validation(
    source_column: Iterable[Iterable[Any]], expected: list[str]
) -> None:
    actual = _format_source(source_column)
    assert actual == expected
