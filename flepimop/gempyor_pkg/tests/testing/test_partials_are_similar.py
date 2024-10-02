from functools import partial
from typing import Literal

import pytest

from gempyor.testing import partials_are_similar


def add_two_numbers(x: int | float, y: int | float) -> float:
    return float(x) + float(y)


def combine_two_numbers(
    x: int | float, y: int | float, how: Literal["sum", "product"] = "sum"
) -> float:
    if how == "sum":
        return float(x) + float(y)
    return float(x) * float(y)


class TestPartialsAreSimilar:
    @pytest.mark.parametrize(
        "f,g,check_func,check_args,check_keywords",
        [
            (
                partial(add_two_numbers, 2),
                partial(add_two_numbers, 2),
                True,
                True,
                True,
            ),
            (
                partial(add_two_numbers, 2),
                partial(add_two_numbers, 2.0),
                True,
                True,
                True,
            ),
            (
                partial(add_two_numbers, 2),
                partial(add_two_numbers, 3),
                True,
                False,
                True,
            ),
            (
                partial(add_two_numbers, 2),
                partial(add_two_numbers, 3.0),
                True,
                False,
                True,
            ),
            (
                partial(add_two_numbers, 2.0),
                partial(combine_two_numbers, 2.0),
                False,
                True,
                True,
            ),
            (
                partial(add_two_numbers, 2.0),
                partial(combine_two_numbers, 3.0),
                False,
                False,
                True,
            ),
            (
                partial(add_two_numbers, 2.0),
                partial(combine_two_numbers, 2.0, how="product"),
                False,
                True,
                False,
            ),
            (
                partial(combine_two_numbers, 2, how="sum"),
                partial(combine_two_numbers, 2, how="product"),
                True,
                True,
                False,
            ),
            (
                partial(combine_two_numbers, 2, how="sum"),
                partial(combine_two_numbers, 2.0, how="product"),
                True,
                True,
                False,
            ),
            (
                partial(combine_two_numbers, 2),
                partial(combine_two_numbers, 2, how="sum"),
                True,
                True,
                False,
            ),
        ],
    )
    def test_output_validation(
        self,
        f: partial,
        g: partial,
        check_func: bool,
        check_args: bool,
        check_keywords: bool,
    ) -> None:
        assert f != g
        assert partials_are_similar(
            f,
            g,
            check_func=check_func,
            check_args=check_args,
            check_keywords=check_keywords,
        )
