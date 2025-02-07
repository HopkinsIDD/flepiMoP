# Exports
__all__ = ("Seeding", "SeedingFactory")


# Imports
from datetime import date
import logging
from typing import Any
import warnings

import confuse
import numba as nb
import numpy as np
import numpy.typing as npt
import pandas as pd

from .compartments import Compartments
from .simulation_component import SimulationComponent
from .subpopulation_structure import SubpopulationStructure
from . import utils


# Globals
logger = logging.getLogger(__name__)


# Internal functionality
def _DataFrame2NumbaDict(
    df: pd.DataFrame,
    amounts: list[float],
    compartments: Compartments,
    subpop_struct: SubpopulationStructure,
    n_days: int,
    ti: date,
) -> tuple[nb.typed.Dict, npt.NDArray[np.number]]:
    # This functions is extremely unsafe and should only be used after the dataframe has
    # been filtered on dates and subpop according to the limits sets in `modinf`. And
    # sorted by date.
    if not df["date"].is_monotonic_increasing:
        raise ValueError("The `df` given is not sorted by the 'date' column.")

    cmp_grp_names = [col for col in compartments.compartments.columns if col != "name"]
    seeding_dict: nb.typed.Dict = nb.typed.Dict.empty(
        key_type=nb.types.unicode_type,
        value_type=nb.types.int64[:],
    )
    seeding_dict["seeding_sources"] = np.zeros(len(amounts), dtype=np.int64)
    seeding_dict["seeding_destinations"] = np.zeros(len(amounts), dtype=np.int64)
    seeding_dict["seeding_subpops"] = np.zeros(len(amounts), dtype=np.int64)
    seeding_amounts = np.zeros(len(amounts), dtype=np.float64)

    nb_seed_perday = np.zeros(n_days, dtype=np.int64)

    n_seeding_ignored_before = 0
    n_seeding_ignored_after = 0

    # id_seed = 0
    for idx, (row_index, row) in enumerate(df.iterrows()):
        if row["subpop"] not in subpop_struct.subpop_names:
            logging.debug(
                f"Invalid subpop '{row['subpop']}' in row {row_index + 1} of "
                "seeding::lambda_file. Not found in geodata... Skipping"
            )
        elif (row["date"].date() - ti).days >= 0:
            if (row["date"].date() - ti).days < len(nb_seed_perday):
                nb_seed_perday[(row["date"].date() - ti).days] = (
                    nb_seed_perday[(row["date"].date() - ti).days] + 1
                )
                source_dict = {
                    grp_name: row[f"source_{grp_name}"] for grp_name in cmp_grp_names
                }
                destination_dict = {
                    grp_name: row[f"destination_{grp_name}"] for grp_name in cmp_grp_names
                }
                seeding_dict["seeding_sources"][idx] = compartments.get_comp_idx(
                    source_dict,
                    error_info=(
                        f"(seeding source at idx={idx}, "
                        f"row_index={row_index}, row=>>{row}<<)"
                    ),
                )
                seeding_dict["seeding_destinations"][idx] = compartments.get_comp_idx(
                    destination_dict,
                    error_info=(
                        f"(seeding destination at idx={idx}, "
                        f"row_index={row_index}, row=>>{row}<<)"
                    ),
                )
                seeding_dict["seeding_subpops"][idx] = subpop_struct.subpop_names.index(
                    row["subpop"]
                )
                seeding_amounts[idx] = amounts[idx]
            else:
                n_seeding_ignored_after += 1
        else:
            n_seeding_ignored_before += 1

    if n_seeding_ignored_before > 0:
        logging.critical(
            f"Seeding ignored {n_seeding_ignored_before} rows "
            "because they were before the start of the simulation."
        )
    if n_seeding_ignored_after > 0:
        logging.critical(
            f"Seeding ignored {n_seeding_ignored_after} rows "
            "because they were after the end of the simulation."
        )

    day_start_idx = np.zeros(n_days + 1, dtype=np.int64)
    day_start_idx[1:] = np.cumsum(nb_seed_perday)
    seeding_dict["day_start_idx"] = day_start_idx

    return seeding_dict, seeding_amounts


# Exported functionality
class Seeding(SimulationComponent):
    """
    Class to handle the seeding of the simulation.

    Attributes:
        seeding_config: The configuration for the seeding.
        path_prefix: The path prefix to use when reading files.
    """

    def __init__(self, config: confuse.ConfigView, path_prefix: str = "."):
        """
        Initialize a seeding instance.

        Args:
            config: The configuration for the seeding.
            path_prefix: The path prefix to use when reading files.
        """
        self.seeding_config = config
        self.path_prefix = path_prefix

    def get_from_config(
        self,
        compartments: Compartments,
        subpop_struct: SubpopulationStructure,
        n_days: int,
        ti: date,
        tf: date,
        input_filename: str | None,
    ) -> tuple[nb.typed.Dict, npt.NDArray[np.number]]:
        """
        Get seeding data from the configuration.

        Args:
            compartments: The compartments for the simulation.
            subpop_struct: The subpopulation structure for the simulation.
            n_days: The number of days in the simulation.
            ti: The start date of the simulation.
            tf: The end date of the simulation.
            input_filename: The input filename to use for seeding data. Only used if
                the seeding method is 'FolderDraw'.

        Returns:
            A tuple containing the seeding data as a Numba dictionary and the seeding
            amounts as a Numpy array. The seeding data is a dictionary with the
            following keys:
                - "seeding_sources": The source compartments for the seeding.
                - "seeding_destinations": The destination compartments for the seeding.
                - "seeding_subpops": The subpopulations for the seeding.
                - "day_start_idx": The start index for each day in the seeding data.
        """
        method = "NoSeeding"
        if self.seeding_config is not None and "method" in self.seeding_config.keys():
            method = self.seeding_config["method"].as_str()

        if method == "NegativeBinomialDistributed" or method == "PoissonDistributed":
            seeding = pd.read_csv(
                self.path_prefix / self.seeding_config["lambda_file"].as_str(),
                converters={"subpop": lambda x: str(x)},
                parse_dates=["date"],
                skipinitialspace=True,
            )
            dupes = seeding[seeding.duplicated(["subpop", "date"])].index + 1
            if not dupes.empty:
                raise ValueError(
                    f"There are repeating subpop-date in rows '{dupes.tolist()}' "
                    "of `seeding::lambda_file`."
                )
        elif method == "FolderDraw":
            seeding = pd.read_csv(
                self.path_prefix / input_filename,
                converters={"subpop": lambda x: str(x)},
                parse_dates=["date"],
                skipinitialspace=True,
            )
        elif method == "FromFile":
            seeding = pd.read_csv(
                self.path_prefix / self.seeding_config["seeding_file"].get(),
                converters={"subpop": lambda x: str(x)},
                parse_dates=["date"],
                skipinitialspace=True,
            )
        elif method == "NoSeeding":
            seeding = pd.DataFrame(columns=["date", "subpop"])
            return _DataFrame2NumbaDict(
                seeding, [], compartments, subpop_struct, n_days, ti
            )
        else:
            raise ValueError(f"Unknown seeding method given, '{method}'.")

        # Sorting by date is important for the seeding format
        seeding = seeding.sort_values(by="date", axis="index").reset_index(drop=True)
        mask = (seeding["date"].dt.date > ti) & (seeding["date"].dt.date <= tf)
        seeding = seeding.loc[mask].reset_index(drop=True)
        mask = seeding["subpop"].isin(subpop_struct.subpop_names)
        seeding = seeding.loc[mask].reset_index(drop=True)

        amounts = np.zeros(len(seeding))
        if method == "PoissonDistributed":
            amounts = np.random.poisson(seeding["amount"])
        elif method == "NegativeBinomialDistributed":
            raise ValueError(
                "Seeding method 'NegativeBinomialDistributed' "
                "is not supported by flepiMoP anymore."
            )
        elif method == "FolderDraw" or method == "FromFile":
            amounts = seeding["amount"]

        return _DataFrame2NumbaDict(
            seeding, amounts, compartments, subpop_struct, n_days, ti
        )

    def get_from_file(
        self, *args: Any, **kwargs: Any
    ) -> tuple[nb.typed.Dict, npt.NDArray[np.number]]:
        """
        This method is deprecated. Use `get_from_config` instead.

        Args:
            *args: Positional arguments to pass to `get_from_config`.
            **kwargs: Keyword arguments to pass to `get_from_config`.

        Returns:
            The result of `get_from_config`.
        """
        warnings.warn(
            "The 'get_from_file' method is deprecated. Use 'get_from_config' instead.",
            DeprecationWarning,
        )
        return self.get_from_config(*args, **kwargs)


def SeedingFactory(config: confuse.ConfigView, path_prefix: str = ".") -> Seeding:
    """
    Create a Seeding instance based on the given configuration.

    This function will use the given configuration to either lookup a plugin class for
    the seeding instance or fallback to the default Seeding class.

    Args:
        config: The configuration for the seeding.
        path_prefix: The path prefix to use when reading files.

    Returns:
        A Seeding instance.
    """
    if config is not None and "method" in config.keys():
        if config["method"].as_str() == "plugin":
            klass = utils.search_and_import_plugins_class(
                plugin_file_path=config["plugin_file_path"].as_str(),
                class_name="Seeding",
                config=config,
                path_prefix=path_prefix,
            )
            return klass
    return Seeding(config, path_prefix=path_prefix)
