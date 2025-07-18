"""Utility functions for checking initial conditions in gempyor."""

__all__: tuple[str, ...] = ()

from collections.abc import Callable
from inspect import signature
from typing import Any
import warnings

import numpy as np
import numpy.typing as npt

from gempyor.parameters import Parameters


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


def _inspect_requested_parameters(
    f: Callable[..., Any],
    ignore_args: int,
    pdata: dict[str, dict[str, Any]],
    p_draw: npt.NDArray[np.float64],
) -> list[float | int | npt.NDArray[np.float64 | np.int64]]:
    """
    Inspect a function for requested parameters.

    Args:
        f: The function to inspect the arguments of.
        ignore_args: The number of initial arguments to ignore.
        pdata: A dictionary of parameter data, like the `pdata` attribute of
            :obj:`gempyor.parameters.Parameters`.
        p_draw: A numpy array of parameter draws to extract values from.

    Returns:
        A list of parameter values extracted from `p_draw` that can be passed to the
        given function `f` after ignoring the first `ignore_args` arguments. The
        values are extracted based on the parameter data in `pdata`.
        * 'dist' parameters are expected to be single values.
        * 'ts' parameters are expected to be time series data and will be returned as
            a numpy array.

    Raises:
        ValueError: If the function does not have enough arguments to ignore.
        ValueError: If a requested parameter is not found in `pdata`.
        NotImplementedError: If a requested parameter is not supported, only 'dist' or
            'ts' parameters are currently supported.
    """
    sig = signature(f)
    arg_names = list(sig.parameters.keys())
    if len(arg_names) < ignore_args:
        msg = (
            f"Function '{f.__name__}' does not have enough arguments to ignore "
            f"the first {ignore_args} arguments. It has {len(arg_names)} arguments "
            f"instead. The arguments are: {arg_names}."
        )
        raise ValueError(msg)
    if not (parameter_args := arg_names[ignore_args:]):
        return []
    args = []
    for param_name in parameter_args:
        if (param_data := pdata.get(param_name)) is None:
            msg = (
                f"The requested parameter, '{param_name}', not "
                f"found in the arguments of {f.__name__}. The "
                f"available parameters are: {pdata.keys()}."
            )
            raise ValueError(msg)
        if "dist" in param_data:
            args.append(p_draw[param_data["idx"], 0, 0].item())
            continue
        elif "ts" in param_data:
            args.append(p_draw[param_data["idx"], :, :])
            continue
        msg = (
            f"Parameter '{param_name}' in function '{f.__name__}' is not supported. "
            "Only parameters with 'dist' or 'ts' in their data are currently "
            f"supported. Instead has the following data: {param_data.keys()}."
        )
        raise NotImplementedError(msg)
    return args
