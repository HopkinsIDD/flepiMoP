"""Unit tests for `gempyor.steps_rk4.rk4_integration`."""

from typing import Literal
from unittest.mock import patch

import numba as nb
import numpy as np
import pytest

from gempyor.steps_rk4 import rk4_integration


@pytest.mark.parametrize("method", ("euler", "stochastic", "rk4"))
def test_stochastic_simulation_works_with_legacy_integration_method(
    method: Literal["euler", "stochastic", "rk4"],
) -> None:
    """Test that simulation works all the integration engines."""
    # These inputs are modeled after those produced by `config_sample_2pop.yml`.
    seeding_data = nb.typed.Dict.empty(
        key_type=nb.types.unicode_type,
        value_type=nb.types.int64[:],
    )
    seeding_data["seeding_sources"] = np.zeros(0, dtype=np.int64)
    seeding_data["seeding_destinations"] = np.zeros(0, dtype=np.int64)
    seeding_data["seeding_subpops"] = np.zeros(0, dtype=np.int64)
    seeding_data["day_start_idx"] = np.zeros(214, dtype=np.int64)

    def jit_wraps(*args, **kwargs):
        return nb.jit(*args, **kwargs)

    with patch("gempyor.steps_rk4.jit", wraps=jit_wraps) as jit_patch:
        result = rk4_integration(
            ncompartments=4,
            nspatial_nodes=2,
            ndays=213,
            parameters=np.ones((4, 213, 2)),
            dt=0.25,
            transitions=np.array([[0, 1, 2], [1, 2, 3], [1, 2, 3], [0, 2, 3], [2, 3, 4]]),
            proportion_info=np.array([[0, 1, 2, 3], [1, 2, 3, 4], [0, 0, 0, 0]]),
            transition_sum_compartments=np.array([0, 2, 1, 2]),
            initial_conditions=np.array(
                [
                    [8.995e03, 1.000e03],
                    [5.000e00, 0.000e00],
                    [0.000e00, 0.000e00],
                    [0.000e00, 0.000e00],
                ]
            ),
            seeding_data=seeding_data,
            seeding_amounts=np.array([], dtype=np.float64),
            mobility_data=np.array([100.0, 20.0]),
            mobility_row_indices=np.array([1, 0], dtype=np.int32),
            mobility_data_indices=np.array([0, 1, 2], dtype=np.int32),
            population=np.array([9000, 1000]),
            method=method,
            silent=True,
        )
        assert jit_patch.call_count == (2 if method == "stochastic" else 3)
    assert isinstance(result, tuple)
