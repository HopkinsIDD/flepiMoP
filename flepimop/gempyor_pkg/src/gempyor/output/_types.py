"""Types to represent the model output data structures."""

__all__: tuple[str, ...] = ()

from dataclasses import dataclass
from datetime import date
from typing import Literal

import numpy as np
import numpy.typing as npt

from .._pydantic_ext import _ensure_list


@dataclass(frozen=True)
class ModifierInfoPeriod:
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

    Examples:
        >>> from datetime import date
        >>> from pprint import pprint
        >>> import numpy as np
        >>> from gempyor.output import Chains, ModifierInfo, ModifierInfoPeriod
        >>> rng = np.random.default_rng(12345)
        >>> shape = (4, 30, 2)  # 4 chains, 30 iterations, 2 parameters
        >>> log_probability = rng.lognormal(size=(shape[0], shape[1]))
        >>> samples = rng.normal(size=shape)
        >>> modifiers = [
        ...     ModifierInfo(
        ...         kind="seir",
        ...         name="seasonal_beta",
        ...         subpops=["subpop1"],
        ...         periods=[
        ...             ModifierInfoPeriod(
        ...                 start_date=date(2020, 1, 1),
        ...                 end_date=date(2020, 6, 30),
        ...             ),
        ...             ModifierInfoPeriod(
        ...                 start_date=date(2020, 7, 1),
        ...                 end_date=date(2020, 12, 31),
        ...             ),
        ...         ],
        ...         parameter="beta",
        ...     ),
        ...     ModifierInfo(
        ...         kind="outcome",
        ...         name="hospitalization_rate",
        ...         subpops=["subpop1", "subpop2"],
        ...         periods=[
        ...             ModifierInfoPeriod(
        ...                 start_date=date(2020, 1, 1),
        ...                 end_date=date(2020, 12, 31),
        ...             ),
        ...         ],
        ...         parameter="hosp::probability",
        ...     ),
        ... ]
        >>> chains = Chains(
        ...     shape=shape,
        ...     log_probability=log_probability,
        ...     samples=samples,
        ...     modifiers=modifiers,
        ... )
        >>> chains.shape
        (4, 30, 2)
        >>> chains.log_probability.shape
        (4, 30)
        >>> chains.samples.shape
        (4, 30, 2)
        >>> pprint(chains.modifiers)
        [ModifierInfo(kind='seir',
                      name='seasonal_beta',
                      subpops=['subpop1'],
                      periods=[ModifierInfoPeriod(start_date=datetime.date(2020, 1, 1),
                                                  end_date=datetime.date(2020, 6, 30)),
                               ModifierInfoPeriod(start_date=datetime.date(2020, 7, 1),
                                                  end_date=datetime.date(2020, 12, 31))],
                      parameter='beta'),
         ModifierInfo(kind='outcome',
                      name='hospitalization_rate',
                      subpops=['subpop1', 'subpop2'],
                      periods=[ModifierInfoPeriod(start_date=datetime.date(2020, 1, 1),
                                                  end_date=datetime.date(2020, 12, 31))],
                      parameter='hosp::probability')]
        >>> chains_subset = chains.subset(chains=[0, 1], iterations=[0, 1, 2])
        >>> chains_subset.shape
        (2, 3, 2)
        >>> chains_subset.log_probability.shape
        (2, 3)
        >>> chains_subset.samples.shape
        (2, 3, 2)
        >>> chains.flatten_samples().shape
        (120, 2)

    """

    shape: tuple[int, int, int]
    log_probability: npt.NDArray[np.float64]
    samples: npt.NDArray[np.float64]
    modifiers: list[ModifierInfo]

    def subset(
        self,
        chains: list[int] | int | None = None,
        iterations: list[int] | int | None = None,
    ) -> "Chains":
        """
        Create a new `Chains` instance that is a subset of the current instance.

        Args:
            iterations: A list, or integer for just one, of iteration indices to include
                in the subset. If `None`, include all iterations.
            chains: A list, or integer for just one, of chain indices to include in the
                subset. If `None`, include all chains.

        Returns:
            A new `Chains` instance containing only the specified chains and iterations.

        """
        chains = list(range(self.shape[0])) if chains is None else _ensure_list(chains)
        iterations = (
            list(range(self.shape[1])) if iterations is None else _ensure_list(iterations)
        )
        return Chains(
            shape=(len(chains), len(iterations), self.shape[2]),
            log_probability=self.log_probability[np.ix_(chains, iterations)],
            samples=self.samples[np.ix_(chains, iterations, np.arange(self.shape[2]))],
            modifiers=self.modifiers,
        )

    def flatten_samples(self) -> npt.NDArray[np.float64]:
        """
        Flatten the samples array to 2D.

        Returns:
            A 2D numpy array of shape (n_chains * n_iterations, n_parameters).

        """
        return self.samples.reshape(-1, self.shape[2])
