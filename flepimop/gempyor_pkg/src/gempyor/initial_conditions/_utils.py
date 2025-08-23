"""Utility functions for checking initial conditions in gempyor."""

__all__: tuple[str, ...] = ()


import warnings

import numpy as np
import numpy.typing as npt


def check_population(
    y0: np.ndarray,
    subpop_names: list[str],
    subpop_pop: npt.NDArray[np.int64],
    ignore_population_checks: bool = False,
) -> None:
    """
    Check that the initial conditions match the population sizes in the model.

    Args:
        y0: The initial conditions array where the row dimension corresponds to the
            compartments and the column dimension corresponds to the subpopulations.
        subpop_names: A list of subpopulation names.
        subpop_pop: A numpy array containing the population sizes for each
            subpopulation.
        ignore_population_checks: If `True`, ignore population checks.

    Raises:
        ValueError: If the initial conditions do not match the population sizes and
            `ignore_population_checks` is `False`.
    """
    y0_by_subpopulation = y0.sum(axis=0)
    delta = np.abs(y0_by_subpopulation - subpop_pop)
    if len(indexes := np.where(delta > 1)[0]):
        for idx in indexes:
            warnings.warn(
                f"`subpop_names` '{subpop_names[idx]}' (idx: plx_idx) has a "
                f"population from initial condition of '{y0_by_subpopulation[idx]}' "
                f"while population geodata is '{subpop_pop[idx]}'. "
                f"(absolute difference should be <1, here is '{delta[idx]}')."
            )
        if not ignore_population_checks:
            raise ValueError(
                "Geodata and initial condition do not agree on population size. "
                "Use `ignore_population_checks: True` to ignore."
            )
        warnings.warn(
            "Population mismatch errors ignored because `ignore_population_checks` is "
            "set to `True`. Execution will continue, but this is not recommended.",
            UserWarning,
        )
