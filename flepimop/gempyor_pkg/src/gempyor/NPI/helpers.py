"""Helpers for interacting with and using modifiers."""

__all__ = ("SpatialGroups", "get_spatial_groups", "reduce_parameter")

from typing import TypedDict

import numpy as np
import pandas as pd

from ..utils import _flatten_list_of_lists, _make_list_of_list


class SpatialGroups(TypedDict):
    """
    Modifier spatial groups.

    Attributes:
        grouped: List of lists of subpopulations that share the same modifier value.
        ungrouped: List of subpopulations that have individual modifier values.
    """

    grouped: list[list[str]]
    ungrouped: list[str]


def reduce_parameter(
    parameter: np.ndarray,
    modification: pd.DataFrame | float,
    method: str = "product",
) -> np.ndarray:
    if isinstance(modification, pd.DataFrame):
        modification = modification.T
        modification.index = pd.to_datetime(modification.index.astype(str))
        modification = modification.resample("1D").ffill().to_numpy()  # Type consistency:
    if method == "reduction_product":
        return parameter * (1 - modification)
    elif method == "sum":
        return parameter + modification
    elif method == "product":
        return parameter * modification
    else:
        raise ValueError(f"Unknown method to do NPI reduction, got {method}")


def get_spatial_groups(
    subpopulations: list[str],
    subpopulation_groups: list[list[str]] | list[str] | str | None,
) -> SpatialGroups:
    """
    Get the spatial groupings from a modifier group config.

    Args:
        grp_config: Configuration view containing 'subpop_groups' key.
        affected_subpops: List of subpopulations affected by the modifier.

    Returns:
        A `SpatialGroups` dictionary with 'grouped' and 'ungrouped' keys.

    Examples:
        >>> from gempyor.NPI.helpers import get_spatial_groups
        >>> get_spatial_groups(["A", "B", "C"], None)
        {'grouped': [], 'ungrouped': ['A', 'B', 'C']}
        >>> get_spatial_groups(["A", "B", "C"], [["A", "B"], ["C"]])
        {'grouped': [['A', 'B'], ['C']], 'ungrouped': []}
        >>> get_spatial_groups(["A", "B", "C"], "all")
        {'grouped': [['A', 'B', 'C']], 'ungrouped': []}
        >>> get_spatial_groups(
        ...     ["A", "B", "C", "D", "E", "F"],
        ...     [["A", "B"], [], ["E", "F"]]
        ... )
        {'grouped': [['A', 'B'], ['E', 'F']], 'ungrouped': ['C', 'D']}
    """
    spatial_groups = SpatialGroups(grouped=[], ungrouped=[])
    if subpopulation_groups is None:
        spatial_groups["ungrouped"] = subpopulations
    elif subpopulation_groups == "all":
        spatial_groups["grouped"] = [subpopulations]
    else:
        spatial_groups["grouped"] = [
            subgrp
            for grp in _make_list_of_list(subpopulation_groups)
            if (subgrp := sorted(list(set(grp).intersection(subpopulations))))
        ]
        spatial_groups["ungrouped"] = sorted(
            set(subpopulations) - set(_flatten_list_of_lists(subpopulation_groups))
        )
    return spatial_groups
