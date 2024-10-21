from typing import Any, Iterable

import pandas as pd
import pytest

from gempyor.compartments import _format_nested_iterables, _unformat_nested_iterables


@pytest.mark.parametrize(
    ("x", "sep", "maxsplit", "expected"),
    (
        ([], "_", -1, []),
        ((), "_", -1, []),
        (set(), "_", -1, []),
        (pd.Series(), "_", -1, []),
        (["a_b", "c", "d_e_f"], "_", -1, [["a", "b"], ["c"], ["d", "e", "f"]]),
        (("",), "_", -1, [[""]]),
        (["a_a_a", "x_y_z"], "_", -1, [["a", "a", "a"], ["x", "y", "z"]]),
        (["a-b-c"], "_", -1, [["a-b-c"]]),
        (pd.Series(data=["a_b", "c", "d-e"]), "_", -1, [["a", "b"], ["c"], ["d-e"]]),
        ([], "$", 2, []),
        (["1%*%2%*%3", "4%*%5", "6"], "%*%", -1, [["1", "2", "3"], ["4", "5"], ["6"]]),
        (["1%*%2%*%3", "4%*%5", "6"], "%*%", 0, [["1%*%2%*%3"], ["4%*%5"], ["6"]]),
        (["1%*%2%*%3", "4%*%5", "6"], "%*%", 1, [["1", "2%*%3"], ["4", "5"], ["6", 1]]),
        (
            ["1%*%2%*%3", "4%*%5", "6"],
            "%*%",
            2,
            [["1", "2", "3"], ["4", "5", 1], ["6", 1, 1]],
        ),
        (
            ["1%*%2%*%3", "4%*%5", "6"],
            "%*%",
            3,
            [["1", "2", "3", 1], ["4", "5", 1, 1], ["6", 1, 1, 1]],
        ),
        (
            ["1%*%2%*%3", "4%*%5", "6"],
            "%*%",
            4,
            [["1", "2", "3", 1, 1], ["4", "5", 1, 1, 1], ["6", 1, 1, 1, 1]],
        ),
    ),
)
def test_unformat_nested_iterables_output_validation(
    x: Iterable[Iterable[Any]], sep: str, maxsplit: int, expected: list[list[str | int]]
) -> None:
    actual = _unformat_nested_iterables(x, sep=sep, maxsplit=maxsplit)
    assert actual == expected


@pytest.mark.parametrize(
    ("x", "sep"),
    (
        ([], "_"),
        (["a_b", "c", "d_e_f"], "_"),
        ([""], "_"),
        (["a-b-c", "def"], "_"),
        (["ab_cd_ef", "gh_ij_kl"], "_"),
        (["1%*%2%*%3", "4%*%5"], "%*%"),
        (["a##2##abc", "2.3##xyz"], "##"),
    ),
)
def test_unformat_is_inverse_of_format_when_list(x: list[str], sep: str) -> None:
    assert (
        _format_nested_iterables(_unformat_nested_iterables(x, sep=sep), sep=sep) == x
    )
