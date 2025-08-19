"""
Functionality for handling subpopulation structures.

The `SubpopulationStructure` class is used to represent subpopulation structures. It
contains the subpopulation names, populations, and mobility matrix.
"""

__all__ = ("SubpopulationStructure",)

import logging
from pathlib import Path
from typing import Annotated
import warnings

import confuse
import numpy as np
import numpy.typing as npt
import pandas as pd
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    RootModel,
    computed_field,
    model_validator,
)
import scipy.sparse

from ._pydantic_ext import _ensure_list, _read_and_validate_dataframe
from .file_paths import _regularize_path
from .utils import _duplicate_strings


logger = logging.getLogger(__name__)


def _load_mobility_matrix(
    mobility_file: Path | None,
    nsubpops: int,
    subpop_names: list[str],
    subpop_pop: npt.NDArray[np.int64],
) -> scipy.sparse.csr_matrix:
    """
    Load the mobility matrix from a file.

    Args:
        mobility_file: Path to the mobility file. Must be a txt, csv, or npz file.
        nsubpops: Number of subpopulations.
        subpop_names: List of subpopulation names.
        subpop_pop: Population of each subpopulation.

    Returns:
        The mobility matrix as a sparse matrix.

    Raises:
        ValueError: If the mobility data is not a txt, csv, or npz file.
        ValueError: If the mobility data has the wrong shape.
        ValueError: If the sum of the mobility data across rows exceeds the source
            subpopulation populations.
    """
    if mobility_file is None:
        logging.critical("No mobility matrix specified -- assuming no one moves")
        mobility = scipy.sparse.csr_matrix(np.zeros((nsubpops, nsubpops)), dtype=int)
    elif mobility_file.suffix == ".txt":
        warnings.warn(
            "Mobility files as matrices are not recommended. "
            "Please switch to long form csv files.",
            PendingDeprecationWarning,
        )
        mobility = scipy.sparse.csr_matrix(np.loadtxt(mobility_file), dtype=int)
    elif mobility_file.suffix in {".csv", ".parquet"}:
        kwargs = (
            {"converters": {"ori": str, "dest": str}, "skipinitialspace": True}
            if mobility_file.suffix == ".csv"
            else {}
        )
        mobility_data = _read_and_validate_dataframe(
            mobility_file, model=MobilityFileTable, **kwargs
        )
        nn_dict = {subpop_name: idx for idx, subpop_name in enumerate(subpop_names)}
        mobility_data["ori_idx"] = mobility_data["ori"].apply(nn_dict.__getitem__)
        mobility_data["dest_idx"] = mobility_data["dest"].apply(nn_dict.__getitem__)
        mobility = scipy.sparse.coo_matrix(
            (mobility_data.amount, (mobility_data.ori_idx, mobility_data.dest_idx)),
            shape=(nsubpops, nsubpops),
            dtype=int,
        ).tocsr()
    elif mobility_file.suffix == ".npz":
        mobility = scipy.sparse.load_npz(mobility_file).astype(int).tocsr()
    else:
        raise ValueError(
            "Mobility data must either be either a txt, csv, or npz "
            f"file, but was given mobility file of '{mobility_file}'."
        )
    if mobility.shape != (nsubpops, nsubpops):
        raise ValueError(
            f"Mobility data has shape of {mobility.shape}, but should "
            f"match geodata shape of {(nsubpops, nsubpops)}."
        )
    # Make sure sum of mobility values <= the population of src subpop
    row_idx = np.where(np.asarray(mobility.sum(axis=1)).ravel() > subpop_pop)[0]
    if len(row_idx) > 0:
        subpops_with_mobility_exceeding_pop = {subpop_names[r] for r in row_idx}
        raise ValueError(
            "The following subpopulations have mobility exceeding "
            f"their population: {', '.join(subpops_with_mobility_exceeding_pop)}."
        )
    return mobility


class GeodataFileRow(BaseModel):
    """
    Representation of a row in the geodata file.

    Attributes:
        subpop: Name of the subpopulation.
        population: Population of the subpopulation.
    """

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
        """
        Validate that the subpopulation names are unique.

        Raises:
            ValueError: If there are duplicate subpopulation names in the geodata file.

        Returns:
            The validated instance of `GeodataFileTable`.
        """
        if duplicate_subpops := _duplicate_strings(r.subpop for r in self.root):
            raise ValueError(
                f"The following subpopulation names are duplicated in the "
                f"geodata file: {duplicate_subpops}."
            )
        return self


class MobilityFileRow(BaseModel):
    """
    Representation of a row in the mobility file.

    Attributes:
        ori: Origin subpopulation.
        dest: Destination subpopulation.
        amount: Amount of mobility between origin and destination.
    """

    ori: str
    dest: str
    amount: Annotated[int, Field(ge=0)]

    @model_validator(mode="after")
    def ori_and_dest_are_different(self) -> "MobilityFileRow":
        """
        Validate that the origin and destination subpopulation names are different.

        Raises:
            ValueError: If the origin and destination subpopulation names are the same.

        Returns:
            The validated instance of `MobilityFileRow`.
        """
        if self.ori == self.dest:
            raise ValueError(
                f"Origin and destination subpopulations cannot be the same, '{self.ori}'."
            )
        return self


class MobilityFileTable(RootModel):
    """
    Representation of the mobility file as a table.

    Attributes:
        root: List of rows in the mobility file.
    """

    root: list[MobilityFileRow]

    @model_validator(mode="after")
    def ori_and_dest_are_primary_key(self) -> "MobilityFileTable":
        """
        Validate that the origin and destination subpopulation names are unique.

        Raises:
            ValueError: If there are duplicate origin and destination subpopulation
                names in the mobility file.

        Returns:
            The validated instance of `MobilityFileTable`.
        """
        duplicate_pairs = set()
        pairs = set()
        for row in self.root:
            pair = (row.ori, row.dest)
            if pair in pairs:
                duplicate_pairs.add(pair)
                continue
            pairs.add(pair)
        if duplicate_pairs:
            raise ValueError(
                "The following origin-destination pairs are duplicated "
                f"in the mobility file: {duplicate_pairs}."
            )
        return self


class SubpopulationStructure(BaseModel):
    """
    Data container for representing subpopulation structures.

    Attributes:
        path_prefix: Path prefix for the geodata and mobility files.
        geodata: Path to the geodata file.
        mobility: Path to the mobility file or `None` if not specified.
        selected: List of selected subpopulation names.

    """

    path_prefix: Path | None = None
    geodata: Path
    mobility: Path | None = None
    selected: Annotated[list[str], BeforeValidator(_ensure_list)] = []

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def initialize_subpopulation_structure(self) -> "SubpopulationStructure":
        """
        Initialize the subpopulation structure by loading the geodata and mobility.

        This method reads the geodata and mobility files, validates the data, and
        initializes the subpopulation structure.

        Returns:
            The initialized instance of `SubpopulationStructure`.
        """
        # pylint: disable=attribute-defined-outside-init
        # Regularize paths
        self.geodata = _regularize_path(self.geodata, prefix=self.path_prefix)
        if self.mobility is not None:
            self.mobility = _regularize_path(self.mobility, prefix=self.path_prefix)
        # Read and validate geodata
        kwargs = (
            {"converters": {"subpop": lambda x: str(x).strip()}, "skipinitialspace": True}
            if self.geodata.suffix == ".csv"
            else {}
        )
        self._data = _read_and_validate_dataframe(
            self.geodata, model=GeodataFileTable, **kwargs
        )
        # Extra attributes
        self._nsubpops = len(self._data)
        self._subpop_pop = self._data["population"].to_numpy()
        self._subpop_names = self._data["subpop"].tolist()
        # Load mobility matrix
        self._mobility_matrix = _load_mobility_matrix(
            self.mobility, self._nsubpops, self._subpop_names, self._subpop_pop
        )
        # Apply selected subpopulations
        if self.selected:
            if selected_missing := set(self.selected) - set(self._subpop_names):
                raise ValueError(
                    "The following selected subpopulations are not "
                    f"in the geodata: {','.join(selected_missing)}."
                )
            selected_subpop_indices = [self._subpop_names.index(s) for s in self.selected]
            self._data = self._data.iloc[selected_subpop_indices]
            self._nsubpops = len(self._data)
            self._subpop_pop = self._subpop_pop[selected_subpop_indices]
            self._subpop_names = self.selected
            self._mobility_matrix = self._mobility_matrix[selected_subpop_indices][
                :, selected_subpop_indices
            ]
        # pylint: enable=attribute-defined-outside-init
        return self

    @computed_field
    @property
    def data(self) -> pd.DataFrame:
        """
        Get the geodata attribute as a pandas DataFrame.

        Returns:
            The geodata file as a DataFrame.
        """
        return self._data

    @computed_field
    @property
    def nsubpops(self) -> int:
        """
        Get the number of subpopulations.

        Returns:
            The number of subpopulations.
        """
        return self._nsubpops

    @computed_field
    @property
    def subpop_pop(self) -> npt.NDArray[np.int64]:
        """
        Get the populations of the subpopulations.

        Returns:
            The populations of the subpopulations as a numpy array.
        """
        return self._subpop_pop

    @computed_field
    @property
    def subpop_names(self) -> list[str]:
        """
        Get the names of the subpopulations.

        Returns:
            The names of the subpopulations as a list.
        """
        return self._subpop_names

    @computed_field
    @property
    def mobility_matrix(self) -> scipy.sparse.csr_matrix:
        """
        Get the mobility matrix.

        Returns:
            The mobility matrix as a sparse matrix.
        """
        return self._mobility_matrix

    @classmethod
    def from_confuse_config(
        cls, config: confuse.Subview, path_prefix: Path | None = None
    ) -> "SubpopulationStructure":
        """
        Create a `SubpopulationStructure` instance from a confuse configuration view.

        Args:
            config: A configuration view containing the subpopulation
                configuration.
            path_prefix: The path prefix for the geodata and mobility files or `None` to
                use the current working directory.

        Returns:
            An instance of `SubpopulationStructure`.
        """
        return cls.model_validate(dict(config.get()) | {"path_prefix": path_prefix})
