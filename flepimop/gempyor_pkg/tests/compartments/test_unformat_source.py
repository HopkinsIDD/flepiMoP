from typing import Any, Iterable

import pandas as pd
import pytest

from gempyor.compartments import _format_source, _unformat_source


@pytest.mark.parametrize(
    ("source_column", "expected"),
    (
        ([], []),
        ((), []),
        (set(), []),
        (pd.Series(), []),
        (["a_b", "c", "d_e_f"], [["a", "b"], ["c"], ["d", "e", "f"]]),
        (("",), [[""]]),
        (["a_a_a", "x_y_z"], [["a", "a", "a"], ["x", "y", "z"]]),
        (["a-b-c"], [["a-b-c"]]),
        (pd.Series(data=["a_b", "c", "d-e"]), [["a", "b"], ["c"], ["d-e"]]),
    ),
)
def test_unformat_source_output_validation(
    source_column: Iterable[Iterable[Any]], expected: list[list[str]]
) -> None:
    actual = _unformat_source(source_column)
    assert actual == expected


@pytest.mark.parametrize(
    "source_column",
    ([], ["a_b", "c", "d_e_f"], [""], ["a-b-c", "def"], ["ab_cd_ef", "gh_ij_kl"]),
)
def test_unformat_is_inverse_of_format_when_list(source_column: list[str]) -> None:
    assert _format_source(_unformat_source(source_column)) == source_column
