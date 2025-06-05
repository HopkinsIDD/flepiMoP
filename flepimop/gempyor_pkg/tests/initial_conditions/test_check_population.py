"""Unit tests for the `gempyor.initial_conditions.check_population` function."""

import numpy as np
import numpy.typing as npt
import pytest

from gempyor.initial_conditions import check_population


@pytest.mark.parametrize(
    ("y0", "subpop_names", "subpop_pop"),
    [
        (
            np.array([[100, 200, 300]], dtype=np.float64),
            ["A", "B", "C"],
            np.array([100, 200, 300]),
        ),
        (
            np.array(
                [
                    [10, 20, 30],
                    [40, 50, 60],
                ],
                dtype=np.float64,
            ),
            ["A", "B", "C"],
            np.array([50, 70, 90]),
        ),
        (  # There is some wiggle room in the initial conditions.
            np.array([[1.5, 1.7, 1.9]], dtype=np.float64),
            ["A", "B", "C"],
            np.array([1, 1, 1]),
        ),
    ],
)
@pytest.mark.parametrize("ignore_population_checks", [True, False])
def test_returns_none_with_no_messages_when_valid(
    y0: npt.NDArray[np.float64],
    subpop_names: list[str],
    subpop_pop: npt.NDArray[np.int64],
    ignore_population_checks: bool,
) -> None:
    """Test that the function returns `None` when the population is valid."""
    assert (
        check_population(
            y0, subpop_names, subpop_pop, ignore_population_checks=ignore_population_checks
        )
        is None
    )
