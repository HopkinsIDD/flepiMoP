"""Unit tests for the `_inspect_requested_parameters` helper function."""

from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from gempyor.parameters import _inspect_requested_parameters


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


def test_dist_requested_parameters_given_as_scalar_for_const_row() -> None:
    """Test that 'dist' parameters are returned as scalars."""
    f = lambda x: x
    pdata = {"x": {"idx": 0, "dist": True}}
    p_draw = np.array([[[1.0]]], dtype=np.float64)
    result = _inspect_requested_parameters(f, 0, pdata, p_draw)
    assert isinstance(result[0], float)


def test_dist_requested_parameters_given_as_array_for_non_const_row() -> None:
    """Test that 'dist' parameters are returned as arrays."""
    f = lambda x: x
    pdata = {"x": {"idx": 0, "dist": True}}
    p_draw = np.array([[[1.0, 2.0]]], dtype=np.float64)
    result = _inspect_requested_parameters(f, 0, pdata, p_draw)
    assert isinstance(result[0], np.ndarray)


def test_ts_requested_parameters_given_as_array() -> None:
    """Test that 'ts' parameters are returned as arrays."""
    f = lambda x: x
    pdata = {"x": {"idx": 0, "ts": True}}
    p_draw = np.array([[[1.0]]], dtype=np.float64)
    result = _inspect_requested_parameters(f, 0, pdata, p_draw)
    assert isinstance(result[0], np.ndarray)


@pytest.mark.parametrize(
    ("f", "ignore_args", "pdata", "p_draw", "expected"),
    [
        (
            lambda x, y: x + y,
            0,
            {"x": {"idx": 0, "dist": True}, "y": {"idx": 1, "dist": True}},
            np.array([[[1.0]], [[2.0]]], dtype=np.float64),
            [1.0, 2.0],
        ),
        (
            lambda x, y: x + y,
            1,
            {"y": {"idx": 0, "dist": True}, "z": {"idx": 1, "dist": True}},
            np.array([[[3.0]], [[4.0]]], dtype=np.float64),
            [3.0],
        ),
        (
            lambda a, b, t, u, x, y: (a + b + t + u) * (x + y),
            2,
            {
                "y": {"idx": 0, "ts": True},
                "x": {"idx": 1, "ts": True},
                "u": {"idx": 2, "dist": True},
                "t": {"idx": 3, "dist": True},
            },
            np.array(
                [
                    [[1.0, 2.0], [3.0, 4.0]],
                    [[5.0, 6.0], [7.0, 8.0]],
                    [[1.23, 1.23], [1.23, 1.23]],
                    [[2.34, 2.34], [2.34, 2.34]],
                ],
                dtype=np.float64,
            ),
            [
                2.34,
                1.23,
                np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float64),
                np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
            ],
        ),
        (
            lambda x, y, z: x + y + z,
            0,
            {
                "x": {"idx": 0, "ts": True},
                "y": {"idx": 1, "dist": True},
                "z": {"idx": 2, "dist": True},
            },
            np.array(
                [
                    [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                    [[1.23, 1.23], [1.23, 1.23], [1.23, 1.23]],
                    [[3.45, 6.78], [3.45, 6.78], [3.45, 6.78]],
                ],
                dtype=np.float64,
            ),
            [
                np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64),
                1.23,
                np.array([3.45, 6.78], dtype=np.float64),
            ],
        ),
    ],
)
def test_exact_results_for_select_inputs(
    f: Callable,
    ignore_args: int,
    pdata: dict[str, dict[str, Any]],
    p_draw: npt.NDArray[np.float64],
    expected: list[float | int | npt.NDArray[np.float64 | np.int64]],
) -> None:
    """Test that the function returns the expected results for given inputs."""
    result = _inspect_requested_parameters(f, ignore_args, pdata, p_draw)
    assert len(result) == len(expected)
    assert all(
        [
            (
                np.allclose(result_value, expected_value)
                if isinstance(expected_value, np.ndarray)
                else result_value == expected_value
            )
            for result_value, expected_value in zip(result, expected)
        ]
    )
