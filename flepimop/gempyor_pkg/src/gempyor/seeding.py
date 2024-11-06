from typing import Dict

import numpy as np
import pandas as pd
import confuse
import logging
from .simulation_component import SimulationComponent
from . import utils
import numba as nb
import os

logger = logging.getLogger(__name__)


## TODO: ideally here path_prefix should not be used and all files loaded from modinf


def _DataFrame2NumbaDict(df, amounts, modinf) -> nb.typed.Dict:
    if not df["date"].is_monotonic_increasing:
        raise ValueError(
            "_DataFrame2NumbaDict got an unsorted dataframe, exposing itself to non-sense"
        )

    cmp_grp_names = [
        col for col in modinf.compartments.compartments.columns if col != "name"
    ]
    seeding_dict: nb.typed.Dict = nb.typed.Dict.empty(
        key_type=nb.types.unicode_type,
        value_type=nb.types.int64[:],
    )
    seeding_dict["seeding_sources"] = np.zeros(len(amounts), dtype=np.int64)
    seeding_dict["seeding_destinations"] = np.zeros(len(amounts), dtype=np.int64)
    seeding_dict["seeding_subpops"] = np.zeros(len(amounts), dtype=np.int64)
    seeding_amounts = np.zeros(len(amounts), dtype=np.float64)

    nb_seed_perday = np.zeros(modinf.n_days, dtype=np.int64)

    n_seeding_ignored_before = 0
    n_seeding_ignored_after = 0

    # id_seed = 0
    for idx, (row_index, row) in enumerate(df.iterrows()):
        if row["subpop"] not in modinf.subpop_struct.subpop_names:
            logging.debug(
                f"Invalid subpop '{row['subpop']}' in row {row_index + 1} of seeding::lambda_file. Not found in geodata... Skipping"
            )
        elif (row["date"].date() - modinf.ti).days >= 0:
            if (row["date"].date() - modinf.ti).days < len(nb_seed_perday):
                nb_seed_perday[(row["date"].date() - modinf.ti).days] = (
                    nb_seed_perday[(row["date"].date() - modinf.ti).days] + 1
                )
                source_dict = {
                    grp_name: row[f"source_{grp_name}"] for grp_name in cmp_grp_names
                }
                destination_dict = {
                    grp_name: row[f"destination_{grp_name}"] for grp_name in cmp_grp_names
                }
                seeding_dict["seeding_sources"][idx] = modinf.compartments.get_comp_idx(
                    source_dict,
                    error_info=f"(seeding source at idx={idx}, row_index={row_index}, row=>>{row}<<)",
                )
                seeding_dict["seeding_destinations"][idx] = (
                    modinf.compartments.get_comp_idx(
                        destination_dict,
                        error_info=f"(seeding destination at idx={idx}, row_index={row_index}, row=>>{row}<<)",
                    )
                )
                seeding_dict["seeding_subpops"][idx] = (
                    modinf.subpop_struct.subpop_names.index(row["subpop"])
                )
                seeding_amounts[idx] = amounts[idx]
                # id_seed+=1
            else:
                n_seeding_ignored_after += 1
        else:
            n_seeding_ignored_before += 1

    if n_seeding_ignored_before > 0:
        logging.critical(
            f"Seeding ignored {n_seeding_ignored_before} rows because they were before the start of the simulation."
        )
    if n_seeding_ignored_after > 0:
        logging.critical(
            f"Seeding ignored {n_seeding_ignored_after} rows because they were after the end of the simulation."
        )

    day_start_idx = np.zeros(modinf.n_days + 1, dtype=np.int64)
    day_start_idx[1:] = np.cumsum(nb_seed_perday)
    seeding_dict["day_start_idx"] = day_start_idx

    return seeding_dict, seeding_amounts


class Seeding(SimulationComponent):
    def __init__(self, config: confuse.ConfigView, path_prefix: str = "."):
        self.seeding_config = config
        self.path_prefix = path_prefix

    def get_from_config(self, sim_id: int, modinf) -> nb.typed.Dict:
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
                    f"Repeated subpop-date in rows {dupes.tolist()} of seeding::lambda_file."
                )
        elif method == "FolderDraw":
            seeding = pd.read_csv(
                self.path_prefix
                / modinf.get_input_filename(
                    ftype=modinf.seeding_config["seeding_file_type"].get(),
                    sim_id=sim_id,
                    extension_override="csv",
                ),
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
            return _DataFrame2NumbaDict(df=seeding, amounts=[], modinf=modinf)
        else:
            raise NotImplementedError(f"unknown seeding method [got: {method}]")

        # Sorting by date is very important here for the seeding format necessary !!!!
        # print(seeding.shape)
        seeding = seeding.sort_values(by="date", axis="index").reset_index()
        # print(seeding)
        mask = (seeding["date"].dt.date > modinf.ti) & (
            seeding["date"].dt.date <= modinf.tf
        )
        seeding = seeding.loc[mask].reset_index()
        # print(seeding.shape)
        # print(seeding)

        # TODO: print.

        amounts = np.zeros(len(seeding))
        if method == "PoissonDistributed":
            amounts = np.random.poisson(seeding["amount"])
        elif method == "NegativeBinomialDistributed":
            raise ValueError(
                "Seeding method 'NegativeBinomialDistributed' is not supported by flepiMoP anymore."
            )
        elif method == "FolderDraw" or method == "FromFile":
            amounts = seeding["amount"]
        else:
            raise ValueError(f"Unknown seeding method: {method}")

        return _DataFrame2NumbaDict(df=seeding, amounts=amounts, modinf=modinf)

    def get_from_file(self, sim_id: int, modinf) -> nb.typed.Dict:
        """only difference with draw seeding is that the sim_id is now sim_id2load"""
        return self.get_from_config(sim_id=sim_id, modinf=modinf)


def SeedingFactory(config: confuse.ConfigView, path_prefix: str = "."):
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
