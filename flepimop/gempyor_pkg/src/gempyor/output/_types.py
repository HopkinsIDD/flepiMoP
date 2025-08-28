"""Types to represent the model output data structures."""

__all__: tuple[str, ...] = ()

from dataclasses import dataclass
from datetime import date
from typing import Literal, NamedTuple

import numpy as np
import numpy.typing as npt
import pandas as pd

from .._pydantic_ext import _ensure_list


class ModifiersDataFrames(NamedTuple):
    """
    DataFrames to hold modifier information.

    Attributes:
        snpi: List of DataFrames for SEIR modifiers.
        hnpi: List of DataFrames for outcome modifiers.

    """

    snpi: list[pd.DataFrame]
    hnpi: list[pd.DataFrame]


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
    # pylint: disable=line-too-long
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
        >>> modifiers_dfs = chains.to_modifiers_dataframes()
        >>> len(modifiers_dfs.snpi)
        120
        >>> modifiers_dfs.snpi[0]
            subpop  modifier_name             start_date               end_date parameter     value
        0  subpop1  seasonal_beta  2020-01-01,2020-07-01  2020-06-30,2020-12-31      beta -0.811887
        >>> modifiers_dfs.hnpi[0]
                    subpop         modifier_name  start_date    end_date          parameter     value
        0  subpop1,subpop2  hospitalization_rate  2020-01-01  2020-12-31  hosp::probability -0.025538

    """
    # pylint: enable=line-too-long

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

    def to_modifiers_dataframes(self) -> ModifiersDataFrames:
        """
        Convert the chains to a `ModifiersDataFrames` instance.

        Returns:
            A `ModifiersDataFrames` instance containing DataFrames for SEIR and outcome
            modifiers with their corresponding sampled values.
        """
        # Construct base DataFrames without values
        empty_base_df = pd.DataFrame(
            columns=[
                "subpop",
                "modifier_name",
                "start_date",
                "end_date",
                "parameter",
                "value",
            ]
        )
        snpi_param_idx = []
        hnpi_param_idx = []
        snpi_base = []
        hnpi_base = []
        for i, modifier in enumerate(self.modifiers):
            param_idx = snpi_param_idx if modifier.kind == "seir" else hnpi_param_idx
            base = snpi_base if modifier.kind == "seir" else hnpi_base
            param_idx.append(i)
            base.append(
                {
                    "subpop": ",".join(modifier.subpops),
                    "modifier_name": modifier.name,
                    "start_date": ",".join(
                        [p.start_date.strftime("%Y-%m-%d") for p in modifier.periods]
                    ),
                    "end_date": ",".join(
                        [p.end_date.strftime("%Y-%m-%d") for p in modifier.periods]
                    ),
                    "parameter": modifier.parameter,
                }
            )
        snpi_base_df = (
            pd.DataFrame.from_records(snpi_base) if snpi_base else empty_base_df.copy()
        )
        hnpi_base_df = (
            pd.DataFrame.from_records(hnpi_base) if hnpi_base else empty_base_df.copy()
        )
        # Expand the base DataFrame for each chain and iteration
        nchains, niterations, _ = self.shape
        do_snpi = bool(snpi_param_idx)
        do_hnpi = bool(hnpi_param_idx)
        snpi_dfs = []
        hnpi_dfs = []
        for i in range(nchains):
            for j in range(niterations):
                snpi_df = snpi_base_df.copy()
                if do_snpi:
                    snpi_df["value"] = self.samples[i, j, snpi_param_idx]
                hnpi_df = hnpi_base_df.copy()
                if do_hnpi:
                    hnpi_df["value"] = self.samples[i, j, hnpi_param_idx]
                snpi_dfs.append(snpi_df)
                hnpi_dfs.append(hnpi_df)

        return ModifiersDataFrames(
            snpi=snpi_dfs,
            hnpi=hnpi_dfs,
        )
