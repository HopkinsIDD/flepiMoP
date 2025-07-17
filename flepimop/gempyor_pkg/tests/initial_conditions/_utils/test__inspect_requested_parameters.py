"""Unit tests for the `_inspect_requested_parameters` helper function."""

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from gempyor.initial_conditions._utils import _inspect_requested_parameters


@pytest.mark.parametrize(("f", "ignore_args"), [(lambda: None, 1), (lambda x: x, 2)])
def test_too_few_arguments_to_ignore(f: Callable, ignore_args: int) -> None:
    """Test that an error is raised when too few arguments are provided to ignore."""
    with pytest.raises(
        ValueError,
        match=(
            r"^Function '.*' does not have enough arguments to "
            rf"ignore the first {ignore_args} arguments. It has "
            r"[0-9]+ arguments instead\. The arguments are\: .*\.$"
        ),
    ):
        _inspect_requested_parameters(
            f,
            ignore_args,
            {"y": {"idx": 0, "dist": True}},
            np.ones((1, 1, 1), dtype=np.float64),
        )


@pytest.mark.parametrize(
    ("f", "pdata", "param_name"),
    [
        (lambda x: x, {"y": {"idx": 0, "dist": True}}, "x"),
        (
            lambda x, y: x + y,
            {"x": {"idx": 0, "dist": True}, "z": {"idx": 1, "dist": True}},
            "y",
        ),
        (
            lambda x, y: x + y,
            {"y": {"idx": 0, "dist": True}, "z": {"idx": 1, "dist": True}},
            "x",
        ),
    ],
)
def test_requested_parameter_not_found(
    f: Callable, pdata: dict[str, dict[str, Any]], param_name: str
) -> None:
    """Test that an error is raised when a requested parameter is not found in `pdata`."""
    with pytest.raises(
        ValueError,
        match=(
            rf"^The requested parameter\, \'{param_name}\'\, not "
            r"found in the arguments of .*\. The "
            r"available parameters are\: .*\.$"
        ),
    ):
        _inspect_requested_parameters(
            f, 0, pdata, np.ones((len(pdata), 1, 1), dtype=np.float64)
        )


@pytest.mark.parametrize(
    ("f", "pdata", "param_name"),
    [
        (lambda x: x, {"x": {"idx": 0, "other": True}}, "x"),
        (
            lambda x, y: x + y,
            {"x": {"idx": 0, "dist": True}, "y": {"idx": 1, "other": True}},
            "y",
        ),
        (
            lambda x, y: x + y,
            {"y": {"idx": 0, "dist": True}, "x": {"idx": 1, "other": True}},
            "x",
        ),
    ],
)
def test_only_dist_and_ts_parameters_supported(
    f: Callable, pdata: dict[str, dict[str, Any]], param_name: str
) -> None:
    """Test that an error is raised when a parameter is not supported."""
    with pytest.raises(
        NotImplementedError,
        match=(
            rf"^Parameter \'{param_name}\' in function \'.*\' is not supported\. "
            r"Only parameters with 'dist' or 'ts' in their data are currently "
            r"supported\. Instead has the following data\: .*\.$"
        ),
    ):
        _inspect_requested_parameters(
            f, 0, pdata, np.ones((len(pdata), 1, 1), dtype=np.float64)
        )


def test_dist_requested_parameters_given_as_scalar() -> None:
    """Test that 'dist' parameters are returned as scalars."""
    f = lambda x: x
    pdata = {"x": {"idx": 0, "dist": True}}
    p_draw = np.array([[[1.0]]], dtype=np.float64)
    result = _inspect_requested_parameters(f, 0, pdata, p_draw)
    assert isinstance(result[0], float)


def test_ts_requested_parameters_given_as_array() -> None:
    """Test that 'ts' parameters are returned as arrays."""
    f = lambda x: x
    pdata = {"x": {"idx": 0, "ts": True}}
    p_draw = np.array([[[1.0]]], dtype=np.float64)
    result = _inspect_requested_parameters(f, 0, pdata, p_draw)
    assert isinstance(result[0], np.ndarray)
