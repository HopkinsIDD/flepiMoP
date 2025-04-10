"""
Functionality for handling subpopulation structures.

The `SubpopulationStructure` class is used to represent subpopulation structures. It
contains the subpopulation names, populations, and mobility matrix.
"""

__all__ = ("SUBPOP_NAMES_KEY", "SUBPOP_POP_KEY", "SubpopulationStructure")

import logging
import pathlib
import typing
import warnings

import confuse
import numpy as np
import pandas as pd
import scipy.sparse

from .utils import _duplicate_strings


logger = logging.getLogger(__name__)

SUBPOP_POP_KEY: typing.Final[str] = "population"
SUBPOP_NAMES_KEY: typing.Final[str] = "subpop"


class SubpopulationStructure:
    """
    Data container for representing subpopulation structures.

    Attributes:
        setup_name: Name of the setup
        data: DataFrame with subpopulations and populations
        nsubpops: Number of subpopulations
        subpop_pop: Population of each subpopulation
        subpop_names: Names of each subpopulation
        mobility: Mobility matrix where the row dimension corresponds to the source and
            the column dimension corresponds to the destination subpopulation.
    """

    def __init__(
        self,
        *,
        setup_name: str,
        subpop_config: confuse.Subview,
        path_prefix=pathlib.Path("."),
    ):
        """
        Initialize the a subpopulation structure instance.

        Args:
            setup_name: Name of the setup.
            subpop_config: A configuration view containing the subpopulation
                configuration.
            path_prefix: The path prefix for the geodata and mobility files.

        Raises:
            ValueError: If the geodata file does not contain the 'population' column.
            ValueError: If the geodata file does not contain the 'subpop' column.
            ValueError: If there are subpopulations with zero population.
            ValueError: If there are duplicate subpopulation names.
        """
        self.setup_name = setup_name

        geodata_file = path_prefix / subpop_config["geodata"].get()
        self.data = pd.read_csv(
            geodata_file,
            converters={SUBPOP_NAMES_KEY: lambda x: str(x).strip()},
            skipinitialspace=True,
        )
        self.nsubpops = len(self.data)

        if SUBPOP_POP_KEY not in self.data:
            raise ValueError(
                f"The '{SUBPOP_POP_KEY}' column was not "
                f"found in the geodata file '{geodata_file}'."
            )
        if SUBPOP_NAMES_KEY not in self.data:
            raise ValueError(
                f"The '{SUBPOP_NAMES_KEY}' column was not "
                f"found in the geodata file '{geodata_file}'."
            )

        self.subpop_pop = self.data[SUBPOP_POP_KEY].to_numpy()
        self.subpop_names = self.data[SUBPOP_NAMES_KEY].tolist()

        if (zero_subpops := self.nsubpops - np.count_nonzero(self.subpop_pop)) > 0:
            raise ValueError(f"There are {zero_subpops} subpops with zero population.")

        if duplicate_subpop_names := _duplicate_strings(self.subpop_names):
            raise ValueError(
                "The following subpopulation names are duplicated in the "
                f"geodata file '{geodata_file}': {duplicate_subpop_names}."
            )

        if subpop_config["mobility"].exists():
            self.mobility = self._load_mobility_matrix(
                path_prefix / subpop_config["mobility"].get()
            )
        else:
            logging.critical("No mobility matrix specified -- assuming no one moves")
            self.mobility = scipy.sparse.csr_matrix(
                np.zeros((self.nsubpops, self.nsubpops)), dtype=int
            )

        if subpop_config["selected"].exists():
            selected = subpop_config["selected"].get()
            if not isinstance(selected, list):
                selected = [selected]
            # find the indices of the selected subpopulations
            selected_subpop_indices = [self.subpop_names.index(s) for s in selected]
            # filter all the lists
            self.data = self.data.iloc[selected_subpop_indices]
            self.subpop_pop = self.subpop_pop[selected_subpop_indices]
            self.subpop_names = selected
            self.nsubpops = len(self.data)
            self.mobility = self.mobility[selected_subpop_indices][
                :, selected_subpop_indices
            ]

    def _load_mobility_matrix(self, mobility_file: pathlib.Path) -> scipy.sparse.csr_matrix:
        """
        Load the mobility matrix from a file.

        Args:
            mobility_file: Path to the mobility file. Must be a txt, csv, or npz file.

        Returns:
            The mobility matrix as a sparse matrix.

        Raises:
            ValueError: If the mobility data is not a txt, csv, or npz file.
            ValueError: If the mobility data has the same origin and destination and is
                in csv long form.
            ValueError: If the mobility data has the wrong shape.
            ValueError: If the mobility data has entries that exceed the source
                subpopulation populations.
            ValueError: If the sum of the mobility data across rows exceeds the source
                subpopulation populations.
        """
        if mobility_file.suffix == ".txt":
            warnings.warn(
                "Mobility files as matrices are not recommended. "
                "Please switch to long form csv files.",
                PendingDeprecationWarning,
            )
            mobility = scipy.sparse.csr_matrix(np.loadtxt(mobility_file), dtype=int)
        elif mobility_file.suffix == ".csv":
            mobility_data = pd.read_csv(
                mobility_file,
                converters={"ori": str, "dest": str},
                skipinitialspace=True,
            )
            nn_dict = {v: k for k, v in enumerate(self.subpop_names)}
            mobility_data["ori_idx"] = mobility_data["ori"].apply(nn_dict.__getitem__)
            mobility_data["dest_idx"] = mobility_data["dest"].apply(nn_dict.__getitem__)
            if any(mobility_data["ori_idx"] == mobility_data["dest_idx"]):
                raise ValueError(
                    "Mobility fluxes with same origin and destination "
                    "in long form matrix. This is not supported."
                )
            mobility = scipy.sparse.coo_matrix(
                (mobility_data.amount, (mobility_data.ori_idx, mobility_data.dest_idx)),
                shape=(self.nsubpops, self.nsubpops),
                dtype=int,
            ).tocsr()
        elif mobility_file.suffix == ".npz":
            mobility = scipy.sparse.load_npz(mobility_file).astype(int)
        else:
            raise ValueError(
                "Mobility data must either be either a txt, csv, or npz "
                f"file, but was given mobility file of '{mobility_file}'."
            )

        if mobility.shape != (self.nsubpops, self.nsubpops):
            raise ValueError(
                f"Mobility data has shape of {mobility.shape}, but should "
                f"match geodata shape of {(self.nsubpops, self.nsubpops)}."
            )

        # Make sure sum of mobility values <= the population of src subpop
        row_idx = np.where(np.asarray(mobility.sum(axis=1)).ravel() > self.subpop_pop)[0]
        if len(row_idx) > 0:
            subpops_with_mobility_exceeding_pop = {self.subpop_names[r] for r in row_idx}
            raise ValueError(
                "The following subpopulations have mobility exceeding "
                f"their population: {', '.join(subpops_with_mobility_exceeding_pop)}."
            )

        return mobility
