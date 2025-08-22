"""Types to represent the model output data structures."""

__all__: tuple[str, ...] = ()

from dataclasses import dataclass
from datetime import date
from typing import Literal

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class ModifierInfoPeriod:
    # pylint: disable=line-too-long
    """
    Dataclass to hold information about a modifier period.

    Attributes:
        start_date: The start date of the modifier period.
        end_date: The end date of the modifier period.

    Examples:
        >>> from datetime import date
        >>> from pprint import pprint
        >>> from gempyor.output import ModifierInfoPeriod
        >>> period = ModifierInfoPeriod(
        ...     start_date=date(2020, 1, 1),
        ...     end_date=date(2020, 12, 31),
        ... )
        >>> pprint(period)
        ModifierInfoPeriod(start_date=datetime.date(2020, 1, 1),
                           end_date=datetime.date(2020, 12, 31))
    """
    # pylint: enable=line-too-long

    start_date: date
    end_date: date


@dataclass(frozen=True)
class ModifierInfo:
    """
    Dataclass to hold information about a modifier used in the model.

    Attributes:
        kind: The kind of modifier, either 'seir' or 'outcome'.
        name: The name of the modifier.
        subpops: A list of subpopulation names the modifier applies to.
        start_date: The start date of the modifier.
        end_date: The end date of the modifier.
        parameter: The name of the parameter being modified.

    Examples:
        >>> from datetime import date
        >>> from pprint import pprint
        >>> from gempyor.output import ModifierInfo, ModifierInfoPeriod
        >>> periods = [
        ...     ModifierInfoPeriod(
        ...         start_date=date(2020, 1, 1),
        ...         end_date=date(2020, 1, 31),
        ...     ),
        ...     ModifierInfoPeriod(
        ...         start_date=date(2020, 3, 1),
        ...         end_date=date(2020, 3, 31),
        ...     ),
        ... ]
        >>> pprint(periods)
        [ModifierInfoPeriod(start_date=datetime.date(2020, 1, 1),
                            end_date=datetime.date(2020, 1, 31)),
         ModifierInfoPeriod(start_date=datetime.date(2020, 3, 1),
                            end_date=datetime.date(2020, 3, 31))]
        >>> modifier_info = ModifierInfo(
        ...     kind="seir",
        ...     name="seasonal_gamma",
        ...     subpops=["subpop1", "subpop2"],
        ...     periods=periods,
        ...     parameter="gamma",
        ... )
        >>> pprint(modifier_info)
        ModifierInfo(kind='seir',
                     name='seasonal_gamma',
                     subpops=['subpop1', 'subpop2'],
                     periods=[ModifierInfoPeriod(start_date=datetime.date(2020, 1, 1),
                                                 end_date=datetime.date(2020, 1, 31)),
                              ModifierInfoPeriod(start_date=datetime.date(2020, 3, 1),
                                                 end_date=datetime.date(2020, 3, 31))],
                     parameter='gamma')
    """

    kind: Literal["seir", "outcome"]
    name: str
    subpops: list[str]
    periods: list[ModifierInfoPeriod]
    parameter: str


@dataclass(frozen=True)
class Chains:
    """
    Dataclass to hold the chains of a model output.

    Attributes:
        shape: The shape of the chains represented as a tuple of integers corresponding
            to number of chains, iterations, and parameters.
        log_probability: A 2D numpy array of the evaluated log probabilities for each
            chain and iteration. Has shape (n_chains, n_iterations).
        samples: A 3D numpy array of the sampled values for each chain, iteration,
            and parameter. Has shape (n_chains, n_iterations, n_parameters).
        modifiers: A list of `ModifierInfo` instances describing the modifiers used in
            the model. Corresponds to the order of parameters in the `samples` array.
    """

    shape: tuple[int, int, int]
    log_probability: npt.NDArray[np.float64]
    samples: npt.NDArray[np.float64]
    modifiers: list[ModifierInfo]
