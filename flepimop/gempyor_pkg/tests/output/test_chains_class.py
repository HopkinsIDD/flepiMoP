"""Unit tests for `gempyor.output.Chains` class."""

from datetime import date
from typing import Final

import numpy as np
import pytest
from gempyor.output import Chains, ModifierInfo, ModifierInfoPeriod

rng = np.random.default_rng(12345)


EXAMPLE_CHAINS_ONE: Final[Chains] = Chains(
    shape=(4, 30, 2),
    log_probability=-rng.lognormal(size=(4, 30)),
    samples=rng.normal(size=(4, 30, 2)),
    modifiers=[
        ModifierInfo(
            kind="seir",
            name="seasonal_beta",
            subpops=["subpop1"],
            periods=[
                ModifierInfoPeriod(
                    start_date=date(2020, 1, 1),
                    end_date=date(2020, 6, 30),
                ),
            ],
            parameter="beta",
        ),
        ModifierInfo(
            kind="outcome",
            name="hospitalization_rate",
            subpops=["subpop1", "subpop2"],
            periods=[
                ModifierInfoPeriod(
                    start_date=date(2020, 1, 1),
                    end_date=date(2020, 12, 31),
                ),
            ],
            parameter="hosp::probability",
        ),
    ],
)


def determine_new_shape(selector: list[int] | int | None, previous_shape: int) -> int:
    """Determine the new shape dimension after subsetting."""
    if selector is None:
        return previous_shape
    if isinstance(selector, int):
        return 1
    return len(selector)


@pytest.mark.parametrize(
    ("chains", "which_chains", "which_iterations"),
    [
        (
            EXAMPLE_CHAINS_ONE,
            [0, 2],
            [5, 10, 15],
        ),
        (
            EXAMPLE_CHAINS_ONE,
            [0],
            [5, 10, 15],
        ),
        (
            EXAMPLE_CHAINS_ONE,
            [0, 2, 3],
            [4],
        ),
        (
            EXAMPLE_CHAINS_ONE,
            0,
            0,
        ),
        (
            EXAMPLE_CHAINS_ONE,
            None,
            [0, 1, 2, 3, 4, 5],
        ),
        (
            EXAMPLE_CHAINS_ONE,
            [0, 2],
            None,
        ),
    ],
)
def test_subset_method(
    chains: Chains,
    which_chains: list[int] | int | None,
    which_iterations: list[int] | int | None,
) -> None:
    """Test the `subset` method of the `Chains` class."""
    chains_subset = chains.subset(chains=which_chains, iterations=which_iterations)
    assert isinstance(chains_subset, Chains)
    assert chains.modifiers == chains_subset.modifiers
    assert all(
        new_shape <= old_shape
        for new_shape, old_shape in zip(chains_subset.shape, chains.shape)
    )
    assert chains_subset.shape == (
        determine_new_shape(which_chains, chains.shape[0]),
        determine_new_shape(which_iterations, chains.shape[1]),
        chains.shape[2],
    )
    if isinstance(which_chains, list) and isinstance(which_iterations, list):
        for i, chain_idx in enumerate(which_chains):
            for j, iter_idx in enumerate(which_iterations):
                assert np.all(
                    np.isclose(
                        chains_subset.samples[i, j, :],
                        chains.samples[chain_idx, iter_idx, :],
                    )
                )
                assert np.isclose(
                    chains_subset.log_probability[i, j],
                    chains.log_probability[chain_idx, iter_idx],
                )
