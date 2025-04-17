"""
Functionality for handling subpopulation structures.

The `SubpopulationStructure` class is used to represent subpopulation structures. It
contains the subpopulation names, populations, and mobility matrix.
"""

__all__ = (
    "SubpopulationSetupConfig",
    "SubpopulationStructure",
)

import logging
from pathlib import Path
from typing import Annotated, Final
import warnings

import confuse
import numpy as np
import pandas as pd
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    RootModel,
    model_validator,
)
import scipy.sparse

from ._pydantic_ext import _ensure_list, _read_and_validate_dataframe
from .file_paths import _regularize_path
from .utils import _duplicate_strings


logger = logging.getLogger(__name__)

SUBPOP_POP_KEY: Final[str] = "population"
SUBPOP_NAMES_KEY: Final[str] = "subpop"


class SubpopulationSetupConfig(BaseModel):
    """
    Configuration for subpopulation setup.

    Attributes:
        geodata: Path to the geodata file.
        mobility: Path to the mobility file or `None` for no mobility.
        selected: List of selected subpopulation names or an empty list to use all
            subpopulations provided in `geodata`.
    """

    geodata: Path
    mobility: Path | None = None
    selected: Annotated[list[str], BeforeValidator(_ensure_list)] = []


class GeodataFileRow(BaseModel):
    """
    Representation of a row in the geodata file.

    Attributes:
        subpop: Name of the subpopulation.
        population: Population of the subpopulation.
    """

    model_config = ConfigDict(coerce_numbers_to_str=True)

    subpop: str
    population: Annotated[int, Field(gt=0)]


class GeodataFileTable(RootModel):
    """
    Representation of the geodata file as a table.

    Attributes:
        root: List of rows in the geodata file.
    """

    root: list[GeodataFileRow]

    @model_validator(mode="after")
    def subpop_is_primary_key(self) -> "GeodataFileTable":
        if duplicate_subpops := _duplicate_strings(r.subpop for r in self.root):
            raise ValueError(
                f"The following subpopulation names are duplicated in the "
                f"geodata file: {duplicate_subpops}."
            )
        return self


class SubpopulationStructure:
    """
    Data container for representing subpopulation structures.

    Attributes:
        data: DataFrame with subpopulations and populations
        nsubpops: Number of subpopulations
        subpop_pop: Population of each subpopulation
        subpop_names: Names of each subpopulation
        mobility: Mobility matrix where the row dimension corresponds to the source and
            the column dimension corresponds to the destination subpopulation.
    """

    def __init__(self, subpop_config: confuse.Subview, path_prefix: Path | None = None):
        """
        Initialize the a subpopulation structure instance.

        Args:
            subpop_config: A configuration view containing the subpopulation
                configuration.
            path_prefix: The path prefix for the geodata and mobility files or `None` to
                use the current working directory.

        Raises:
            ValueError: If the geodata file does not contain the 'population' column.
            ValueError: If the geodata file does not contain the 'subpop' column.
            ValueError: If there are subpopulations with zero population.
            ValueError: If there are duplicate subpopulation names.
        """
        self._config = SubpopulationSetupConfig.model_validate(dict(subpop_config.get()))
        self.data = _read_and_validate_dataframe(
            _regularize_path(self._config.geodata, prefix=path_prefix),
            model=GeodataFileTable,
        )
        self.nsubpops = len(self.data)
        self.subpop_pop = self.data["population"].to_numpy()
        self.subpop_names = self.data["subpop"].tolist()
        self.mobility = self._load_mobility_matrix(
            _regularize_path(self._config.mobility, prefix=path_prefix)
        )
        if self._config.selected:
            # find the indices of the selected subpopulations
            selected_subpop_indices = [
                self.subpop_names.index(s) for s in self._config.selected
            ]
            # filter all the lists
            self.data = self.data.iloc[selected_subpop_indices]
            self.subpop_pop = self.subpop_pop[selected_subpop_indices]
            self.subpop_names = self._config.selected
            self.nsubpops = len(self.data)
            self.mobility = self.mobility[selected_subpop_indices][
                :, selected_subpop_indices
            ]

    def _load_mobility_matrix(self, mobility_file: Path | None) -> scipy.sparse.csr_matrix:
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
        if mobility_file is None:
            logging.critical("No mobility matrix specified -- assuming no one moves")
            mobility = scipy.sparse.csr_matrix(
                np.zeros((self.nsubpops, self.nsubpops)), dtype=int
            )
        elif mobility_file.suffix == ".txt":
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
