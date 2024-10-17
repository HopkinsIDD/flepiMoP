from typing import Any, Iterable

import pytest

from gempyor.compartments import _format_source


@pytest.mark.parametrize(
    "source_column", ([[]], [[], [1, 2]], [["x", "y", "z"], [], [1, 2, 3]])
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
        ([[1], [2], [3]], [1, 2, 3]),
        ([[1], [2, 3]], [1, "2_3"]),
        ([[1, 2, 3], [[1, 2]]], ["1_2_3", [1, 2]]),
        ([[1, 2, 3], [[1, 2], [3, 4]]], ["1_2_3", "[1, 2]_[3, 4]"]),
        ([[1, [2, [3]]]], ["1_[2, [3]]"]),
        ([["a"], ["b", "c"], ["d", "e", "f", "g"]], ["a", "b_c", "d_e_f_g"]),
        ([["a", ["b", ["d", ["e"]]]]], ["a_['b', ['d', ['e']]]"]),
        (
            [[["a", 3.14], [True, None], [object, 1]]],
            ["['a', 3.14]_[True, None]_[<class 'object'>, 1]"],
        ),
    ),
)
def test_format_source_output_validation(
    source_column: Iterable[Iterable[Any]], expected: list[str]
) -> None:
    actual = _format_source(source_column)
    assert actual == expected
