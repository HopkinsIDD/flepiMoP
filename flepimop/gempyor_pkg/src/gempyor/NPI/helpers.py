"""Helpers for interacting with and using modifiers."""

__all__ = ("SpatialGroups", "reduce_parameter")

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from ..utils import _flatten_list_of_lists, _make_list_of_list


@dataclass(frozen=True)
class SpatialGroups:
    """
    Modifier spatial groups.

    Attributes:
        grouped: List of lists of subpopulations that share the same modifier value.
        ungrouped: List of subpopulations that have individual modifier values.

    Examples:
        >>> from gempyor.NPI.helpers import SpatialGroups
        >>> sp = SpatialGroups(grouped=(("A", "B"), ("C",)), ungrouped=("D", "E"))
        >>> sp.grouped
        (('A', 'B'), ('C',))
        >>> sp.ungrouped
        ('D', 'E')
        >>> for kind, group in sp:
        ...     print(f"{kind}: {group}")
        ungrouped: ('D',)
        ungrouped: ('E',)
        grouped: ('A', 'B')
        grouped: ('C',)

    """

    grouped: tuple[tuple[str]] = field(default_factory=tuple)
    ungrouped: tuple[str] = field(default_factory=tuple)

    def __iter__(self) -> Iterator[tuple[Literal["grouped", "ungrouped"], tuple[str]]]:
        yield from zip(len(self.ungrouped) * ("ungrouped",), ((g,) for g in self.ungrouped))
        yield from zip(len(self.grouped) * ("grouped",), self.grouped)

    @classmethod
    def from_dict(
        cls, x: dict[str, Sequence[str] | Sequence[Sequence[str]]]
    ) -> "SpatialGroups":
        # pylint: disable=line-too-long
        """
        Create a `SpatialGroups` instance from a dictionary.

        Args:
            x: Dictionary with 'grouped' and 'ungrouped' keys. If a key is missing, it
                defaults to an empty tuple.

        Returns:
            A `SpatialGroups` instance.

        Raises:
            TypeError: If the value for 'grouped' is not a sequence of sequences of
                strings or strings
            TypeError: If the value for 'ungrouped' is not a sequence of strings.

        Examples:
            >>> from gempyor.NPI.helpers import SpatialGroups
            >>> SpatialGroups.from_dict(
            ...     {"grouped": [["A", "B"], ["C"]], "ungrouped": ["D", "E"]}
            ... )
            SpatialGroups(grouped=(('A', 'B'), ('C',)), ungrouped=('D', 'E'))
            >>> SpatialGroups.from_dict({"ungrouped": ["D", "E"]})
            SpatialGroups(grouped=(), ungrouped=('D', 'E'))
            >>> SpatialGroups.from_dict({"grouped": [["A", "B"], ["C"]]})
            SpatialGroups(grouped=(('A', 'B'), ('C',)), ungrouped=())
            >>> SpatialGroups.from_dict({})
            SpatialGroups(grouped=(), ungrouped=())
            >>> SpatialGroups.from_dict({"grouped": "AB"})
            Traceback (most recent call last):
                ...
            TypeError: The 'grouped' key is type <class 'str'>, expected a sequence of sequences of strings or strings. The value was grouped='AB'.
            >>> SpatialGroups.from_dict({"ungrouped": "DE"})
            Traceback (most recent call last):
                ...
            TypeError: The 'ungrouped' key is type <class 'str'>, expected a sequence of strings. The value was ungrouped='DE'.

        """
        # pylint: enable=line-too-long
        # Preliminary type checks
        grouped = x.get("grouped", ())
        if not (
            (isinstance(grouped, Sequence) and not isinstance(grouped, str))
            and all(isinstance(g, (Sequence, str)) for g in grouped)
            and all(
                isinstance(s, str)
                for g in grouped
                for s in (g if isinstance(g, Sequence) else [g])
            )
        ):
            msg = (
                f"The 'grouped' key is type {type(grouped)}, expected a sequence "
                f"of sequences of strings or strings. The value was {grouped=}."
            )
            raise TypeError(msg)
        ungrouped = x.get("ungrouped", ())
        if not (
            (isinstance(ungrouped, Sequence) and not isinstance(ungrouped, str))
            and all(isinstance(s, str) for s in ungrouped)
        ):
            msg = (
                f"The 'ungrouped' key is type {type(ungrouped)}, expected a sequence "
                f"of strings. The value was {ungrouped=}."
            )
            raise TypeError(msg)
        # Convert to tuples for immutability
        return cls(
            grouped=tuple(
                tuple(sorted(g)) if isinstance(g, Sequence) else (g,) for g in grouped
            ),
            ungrouped=tuple(sorted(ungrouped)),
        )

    @classmethod
    def from_subpopulations(
        cls,
        subpopulations: list[str],
        subpopulation_groups: list[list[str]] | list[str] | str | None,
    ) -> "SpatialGroups":
        """
        Get the spatial groupings from a modifier group config.

        Args:
            grp_config: Configuration view containing 'subpop_groups' key.
            affected_subpops: List of subpopulations affected by the modifier.

        Returns:
            A `SpatialGroups` instance constructed from modifier subpopulations.

        Examples:
            >>> from gempyor.NPI.helpers import SpatialGroups
            >>> SpatialGroups.from_subpopulations(["A", "B", "C"], None)
            SpatialGroups(grouped=(), ungrouped=('A', 'B', 'C'))
            >>> SpatialGroups.from_subpopulations(["A", "B", "C"], [["A", "B"], ["C"]])
            SpatialGroups(grouped=(('A', 'B'), ('C',)), ungrouped=())
            >>> SpatialGroups.from_subpopulations(["A", "B", "C"], "all")
            SpatialGroups(grouped=(('A', 'B', 'C'),), ungrouped=())
            >>> SpatialGroups.from_subpopulations(
            ...     ["A", "B", "C", "D", "E", "F"],
            ...     [["A", "B"], [], ["E", "F"]],
            ... )
            SpatialGroups(grouped=(('A', 'B'), ('E', 'F')), ungrouped=('C', 'D'))

        """
        spatial_groups = {}
        if subpopulation_groups is None:
            spatial_groups["ungrouped"] = subpopulations
        elif subpopulation_groups == "all":
            spatial_groups["grouped"] = [subpopulations]
        else:
            spatial_groups["grouped"] = [
                subgrp
                for grp in _make_list_of_list(subpopulation_groups)
                if (subgrp := list(set(grp).intersection(subpopulations)))
            ]
            spatial_groups["ungrouped"] = list(
                set(subpopulations) - set(_flatten_list_of_lists(subpopulation_groups))
            )
        return SpatialGroups.from_dict(spatial_groups)


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
