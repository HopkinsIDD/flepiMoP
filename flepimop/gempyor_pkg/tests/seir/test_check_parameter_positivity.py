"""Unit tests for the `gempyor.seir.check_parameter_positivity` function."""

import re
from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

from gempyor.seir import check_parameter_positivity


def single_value_that_is_negative() -> (
    tuple[npt.NDArray[np.float64], list[str], pd.DatetimeIndex, list[str], str, str, str]
):
    """Case with a single value that is negative."""
    return (
        np.array([[[-1.0]]]),
        ["param1"],
        pd.date_range("2020-01-01", periods=1),
        ["subpop1"],
        "param1",
        "subpop1",
        "2020-01-01",
    )


def single_negative_value() -> (
    tuple[npt.NDArray[np.float64], list[str], pd.DatetimeIndex, list[str], str, str, str]
):
    """Case with a single negative value in a larger array."""
    parsed_parameters = np.ones((7, 8, 9), dtype=np.float64)
    parsed_parameters[4, 5, 6] = -1.0
    return (
        parsed_parameters,
        [f"param{i}" for i in range(7)],
        pd.date_range("2020-01-01", periods=8),
        [f"subpop{i}" for i in range(9)],
        "param4",
        "subpop6",
        "2020-01-06",
    )


def two_negative_values() -> (
    tuple[npt.NDArray[np.float64], list[str], pd.DatetimeIndex, list[str], str, str, str]
):
    """Case with two negative values in different subpops/params."""
    (
        parsed_parameters,
        parameter_names,
        dates,
        subpop_names,
        neg_params,
        neg_subpops,
        first_neg_date,
    ) = single_negative_value()
    parsed_parameters[5, 5, 7] = -1.0
    return (
        parsed_parameters,
        parameter_names,
        dates,
        subpop_names,
        neg_params + ", param5",
        neg_subpops + ", subpop7",
        first_neg_date,
    )


def multiple_date_negative_values() -> (
    tuple[npt.NDArray[np.float64], list[str], pd.DatetimeIndex, list[str], str, str, str]
):
    """Case with multiple negative values in different dates, single subpop/param."""
    parsed_parameters = np.ones((7, 8, 9), dtype=np.float64)
    parsed_parameters[3, 4, 7] = -1.0
    parsed_parameters[3, 5, 7] = -1.0
    parsed_parameters[3, 6, 7] = -1.0
    parsed_parameters[3, 7, 7] = -1.0
    return (
        parsed_parameters,
        [f"param{i}" for i in range(7)],
        pd.date_range("2020-01-01", periods=8),
        [f"subpop{i}" for i in range(9)],
        "param3",
        "subpop7",
        "2020-01-05",
    )


def many_negative_values() -> (
    tuple[npt.NDArray[np.float64], list[str], pd.DatetimeIndex, list[str], str, str, str]
):
    """Case with many negative values in different dates/subpops/params."""
    parsed_parameters = np.ones((7, 8, 9), dtype=np.float64)
    parsed_parameters[3, 4, 6] = -1.0
    parsed_parameters[3, 5, 7] = -1.0
    parsed_parameters[3, 6, 8] = -1.0
    parsed_parameters[3, 7, 6] = -1.0
    parsed_parameters[4, 4, 7] = -1.0
    parsed_parameters[4, 5, 8] = -1.0
    return (
        parsed_parameters,
        [f"param{i}" for i in range(7)],
        pd.date_range("2020-01-01", periods=8),
        [f"subpop{i}" for i in range(9)],
        "param3, param4",
        "subpop5, subpop6, subpop7",
        "2020-01-05",
    )


@pytest.mark.parametrize(
    ("parsed_parameters", "parameter_names", "dates", "subpop_names"),
    [
        (
            np.ones((1, 1, 1), dtype=np.float64),
            ["param1"],
            pd.date_range("2020-01-01", periods=1),
            ["subpop1"],
        ),
        (
            np.ones((3, 4, 5), dtype=np.float64),
            [f"param{i}" for i in range(3)],
            pd.date_range("2020-01-01", periods=4),
            [f"subpop{i}" for i in range(5)],
        ),
        (
            np.random.rand(8, 7, 6),
            [f"param{i}" for i in range(8)],
            pd.date_range("2020-01-01", periods=7),
            [f"subpop{i}" for i in range(6)],
        ),
    ],
)
def test_no_exception_when_parameters_are_not_negative(
    parsed_parameters: npt.NDArray[np.float64],
    parameter_names: list[str],
    dates: pd.DatetimeIndex,
    subpop_names: list[str],
) -> None:
    """Test that no exception is raised when all parameters are non-negative."""
    assert np.all(parsed_parameters >= 0)
    assert (
        check_parameter_positivity(parsed_parameters, parameter_names, dates, subpop_names)
        is None
    )


@pytest.mark.parametrize(
    "negative_input_creator",
    [
        single_value_that_is_negative,
        single_negative_value,
        two_negative_values,
        multiple_date_negative_values,
    ],
)
def test_negative_parameters_raises_value_error(
    negative_input_creator: Callable[
        [],
        tuple[
            npt.NDArray[np.float64], list[str], pd.DatetimeIndex, list[str], str, str, str
        ],
    ],
) -> None:
    """Test that a `ValueError` is raised when there are negative parameters."""
    (
        parsed_parameters,
        parameter_names,
        dates,
        subpop_names,
        neg_params,
        neg_subpops,
        first_neg_date,
    ) = negative_input_creator()
    assert np.any(parsed_parameters < 0)
    with pytest.raises(
        ValueError,
        match=(
            f"There are negative parameter errors in subpops {neg_subpops}, "
            f"starting from date {first_neg_date} in parameters {neg_params}."
        ),
    ):
        check_parameter_positivity(parsed_parameters, parameter_names, dates, subpop_names)
