import pathlib
from typing import Dict, Any, Union

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from numba.typed import Dict
from . import file_paths
import confuse
import logging
from . import compartments
from . import setup
import numba as nb
from .utils import read_df

logger = logging.getLogger(__name__)


def _DataFrame2NumbaDict(df, amounts, setup) -> nb.typed.Dict:
    if not df["date"].is_monotonic_increasing:
        raise ValueError("_DataFrame2NumbaDict got an unsorted dataframe, exposing itself to non-sense")

    cmp_grp_names = [col for col in setup.compartments.compartments.columns if col != "name"]
    seeding_dict: nb.typed.Dict = nb.typed.Dict.empty(
        key_type=nb.types.unicode_type,
        value_type=nb.types.int64[:],
    )
    seeding_dict["seeding_sources"] = np.zeros(len(amounts), dtype=np.int64)
    seeding_dict["seeding_destinations"] = np.zeros(len(amounts), dtype=np.int64)
    seeding_dict["seeding_subpops"] = np.zeros(len(amounts), dtype=np.int64)
    seeding_amounts = np.zeros(len(amounts), dtype=np.float64)

    nb_seed_perday = np.zeros(setup.n_days, dtype=np.int64)

    n_seeding_ignored_before = 0
    n_seeding_ignored_after = 0
    for idx, (row_index, row) in enumerate(df.iterrows()):
        if row["subpop"] not in setup.subpop_struct.subpop_names:
            raise ValueError(
                f"Invalid subpop '{row['subpop']}' in row {row_index + 1} of seeding::lambda_file. Not found in geodata."
            )

        if (row["date"].date() - setup.ti).days >= 0:
            if (row["date"].date() - setup.ti).days < len(nb_seed_perday):
                nb_seed_perday[(row["date"].date() - setup.ti).days] = (
                    nb_seed_perday[(row["date"].date() - setup.ti).days] + 1
                )
                source_dict = {grp_name: row[f"source_{grp_name}"] for grp_name in cmp_grp_names}
                destination_dict = {grp_name: row[f"destination_{grp_name}"] for grp_name in cmp_grp_names}
                seeding_dict["seeding_sources"][idx] = setup.compartments.get_comp_idx(source_dict)
                seeding_dict["seeding_destinations"][idx] = setup.compartments.get_comp_idx(destination_dict)
                seeding_dict["seeding_subpops"][idx] = setup.subpop_struct.subpop_names.index(row["subpop"])
                seeding_amounts[idx] = amounts[idx]
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

    day_start_idx = np.zeros(setup.n_days + 1, dtype=np.int64)
    day_start_idx[1:] = np.cumsum(nb_seed_perday)
    seeding_dict["day_start_idx"] = day_start_idx

    return seeding_dict, seeding_amounts


class SeedingAndIC:
    def __init__(
        self,
        seeding_config: confuse.ConfigView,
        initial_conditions_config: confuse.ConfigView,
    ):
        self.seeding_config = seeding_config
        self.initial_conditions_config = initial_conditions_config

    def draw_ic(self, sim_id: int, setup) -> np.ndarray:
        method = "Default"
        if "method" in self.initial_conditions_config.keys():
            method = self.initial_conditions_config["method"].as_str()

        allow_missing_nodes = False
        allow_missing_compartments = False
        if "allow_missing_nodes" in self.initial_conditions_config.keys():
            if self.initial_conditions_config["allow_missing_nodes"].get():
                allow_missing_nodes = True
        if "allow_missing_compartments" in self.initial_conditions_config.keys():
            if self.initial_conditions_config["allow_missing_compartments"].get():
                allow_missing_compartments = True

        # Places to allocate the rest of the population
        rests = []

        if method == "Default":
            ## JK : This could be specified in the config
            y0 = np.zeros((setup.compartments.compartments.shape[0], setup.nnodes))
            y0[0, :] = setup.popnodes

        elif method == "SetInitialConditions" or method == "SetInitialConditionsFolderDraw":
            #  TODO Think about     - Does not support the new way of doing compartment indexing
            if method == "SetInitialConditionsFolderDraw":
                ic_df = setup.read_simID(ftype=self.initial_conditions_config["initial_file_type"], sim_id=sim_id)
            else:
                ic_df = read_df(
                    self.initial_conditions_config["initial_conditions_file"].get(),
                )

            y0 = np.zeros((setup.compartments.compartments.shape[0], setup.nnodes))
            for pl_idx, pl in enumerate(setup.subpop_struct.subpop_names):  #
                if pl in list(ic_df["subpop"]):
                    states_pl = ic_df[ic_df["subpop"] == pl]
                    for comp_idx, comp_name in setup.compartments.compartments["name"].items():

                        if "mc_name" in states_pl.columns:
                            ic_df_compartment_val = states_pl[states_pl["mc_name"] == comp_name]["amount"]
                        else:
                            filters = setup.compartments.compartments.iloc[comp_idx].drop("name")
                            ic_df_compartment = states_pl.copy()
                            for mc_name, mc_value in filters.items():
                                ic_df_compartment = ic_df_compartment[ic_df_compartment["mc_" + mc_name] == mc_value][
                                    "amount"
                                ]
                        if len(ic_df_compartment_val) > 1:
                            raise ValueError(
                                f"ERROR: Several ({len(ic_df_compartment_val)}) rows are matches for compartment {comp_name} in init file: filters returned {ic_df_compartment_val}"
                            )
                        elif ic_df_compartment_val.empty:
                            if allow_missing_compartments:
                                ic_df_compartment_val = 0.0
                            else:
                                raise ValueError(
                                    f"Initial Conditions: Could not set compartment {comp_name} (id: {comp_idx}) in node {pl} (id: {pl_idx}). The data from the init file is {states_pl}. \n \
                                                 Use 'allow_missing_compartments' to default to 0 for compartments without initial conditions"
                                )
                        if "rest" in ic_df_compartment_val:
                            rests.append([comp_idx, pl_idx])
                        else:
                            y0[comp_idx, pl_idx] = float(ic_df_compartment_val)
                elif allow_missing_nodes:
                    logger.critical(
                        f"No initial conditions for for node {pl}, assuming everyone (n={setup.popnodes[pl_idx]}) in the first metacompartment ({setup.compartments.compartments['name'].iloc[0]})"
                    )
                    if "proportional" in self.initial_conditions_config.keys():
                        if self.initial_conditions_config["proportional"].get():
                            y0[0, pl_idx] = 1.0
                        else:
                            y0[0, pl_idx] = setup.popnodes[pl_idx]
                    else:
                        y0[0, pl_idx] = setup.popnodes[pl_idx]
                else:
                    raise ValueError(
                        f"subpop {pl} does not exist in initial_conditions::states_file. You can set allow_missing_nodes=TRUE to bypass this error"
                    )
        elif method == "InitialConditionsFolderDraw" or method == "FromFile":
            if method == "InitialConditionsFolderDraw":
                ic_df = setup.read_simID(ftype=self.initial_conditions_config["initial_file_type"], sim_id=sim_id)
            elif method == "FromFile":
                ic_df = read_df(
                    self.initial_conditions_config["initial_conditions_file"].get(),
                )

            # annoying conversion because sometime the parquet columns get attributed a timezone...
            ic_df["date"] = pd.to_datetime(ic_df["date"], utc=True)  # force date to be UTC
            ic_df["date"] = ic_df["date"].dt.date
            ic_df["date"] = ic_df["date"].astype(str)

            ic_df = ic_df[(ic_df["date"] == str(setup.ti)) & (ic_df["mc_value_type"] == "prevalence")]
            if ic_df.empty:
                raise ValueError(
                    f"There is no entry for initial time ti in the provided initial_conditions::states_file."
                )
            y0 = np.zeros((setup.compartments.compartments.shape[0], setup.nnodes))

            for comp_idx, comp_name in setup.compartments.compartments["name"].items():
                # rely on all the mc's instead of mc_name to avoid errors due to e.g order.
                # before: only
                # ic_df_compartment = ic_df[ic_df["mc_name"] == comp_name]
                filters = setup.compartments.compartments.iloc[comp_idx].drop("name")
                ic_df_compartment = ic_df.copy()
                for mc_name, mc_value in filters.items():
                    ic_df_compartment = ic_df_compartment[ic_df_compartment["mc_" + mc_name] == mc_value]

                if len(ic_df_compartment) > 1:
                    # ic_df_compartment = ic_df_compartment.iloc[0]
                    raise ValueError(
                        f"ERROR: Several ({len(ic_df_compartment)}) rows are matches for compartment {mc_name} in init file: filter {filters} returned {ic_df_compartment}"
                    )
                elif ic_df_compartment.empty:
                    if allow_missing_compartments:
                        ic_df_compartment = pd.DataFrame(0, columns=ic_df_compartment.columns, index=[0])
                    else:
                        raise ValueError(
                            f"Initial Conditions: Could not set compartment {comp_name} (id: {comp_idx}) in node {pl} (id: {pl_idx}). The data from the init file is {ic_df_compartment[pl]}."
                        )
                elif ic_df_compartment["mc_name"].iloc[0] != comp_name:
                    print(
                        f"WARNING: init file mc_name {ic_df_compartment['mc_name'].iloc[0]} does not match compartment mc_name {comp_name}"
                    )

                for pl_idx, pl in enumerate(setup.subpop_struct.subpop_names):
                    if pl in ic_df.columns:
                        y0[comp_idx, pl_idx] = float(ic_df_compartment[pl])
                    elif allow_missing_nodes:
                        logger.critical(
                            f"No initial conditions for for node {pl}, assuming everyone (n={setup.popnodes[pl_idx]}) in the first metacompartments ({setup.compartments.compartments['name'].iloc[0]})"
                        )
                        if "proportion" in self.initial_conditions_config.keys():
                            if self.initial_conditions_config["proportion"].get():
                                y0[0, pl_idx] = 1.0
                        y0[0, pl_idx] = setup.popnodes[pl_idx]
                    else:
                        raise ValueError(
                            f"subpop {pl} does not exist in initial_conditions::states_file. You can set allow_missing_nodes=TRUE to bypass this error"
                        )
        else:
            raise NotImplementedError(f"unknown initial conditions method [got: {method}]")

        # rest
        if rests:  # not empty
            for comp_idx, pl_idx in rests:
                total = setup.popnodes[pl_idx]
                if "proportional" in self.initial_conditions_config.keys():
                    if self.initial_conditions_config["proportional"].get():
                        total = 1.0
                y0[comp_idx, pl_idx] = total - y0[:, pl_idx].sum()

        if "proportional" in self.initial_conditions_config.keys():
            if self.initial_conditions_config["proportional"].get():
                y0 = y0 * setup.popnodes[pl_idx]

        # check that the inputed values sums to the node_population:
        error = False
        for pl_idx, pl in enumerate(setup.subpop_struct.subpop_names):
            n_y0 = y0[:, pl_idx].sum()
            n_pop = setup.popnodes[pl_idx]
            if abs(n_y0 - n_pop) > 1:
                error = True
                print(
                    f"ERROR: subpop_names {pl} (idx: pl_idx) has a population from initial condition of {n_y0} while population from geodata is {n_pop} (absolute difference should be < 1, here is {abs(n_y0-n_pop)})"
                )
        if error:
            raise ValueError()
        return y0

    def draw_seeding(self, sim_id: int, setup) -> nb.typed.Dict:
        method = "NoSeeding"
        if "method" in self.seeding_config.keys():
            method = self.seeding_config["method"].as_str()

        if method == "NegativeBinomialDistributed" or method == "PoissonDistributed":
            seeding = pd.read_csv(
                self.seeding_config["lambda_file"].as_str(),
                converters={"subpop": lambda x: str(x)},
                parse_dates=["date"],
                skipinitialspace=True,
            )
            dupes = seeding[seeding.duplicated(["subpop", "date"])].index + 1
            if not dupes.empty:
                raise ValueError(f"Repeated subpop-date in rows {dupes.tolist()} of seeding::lambda_file.")
        elif method == "FolderDraw":
            seeding = pd.read_csv(
                setup.get_input_filename(
                    ftype=setup.seeding_config["seeding_file_type"],
                    sim_id=sim_id,
                    extension_override="csv",
                ),
                converters={"subpop": lambda x: str(x)},
                parse_dates=["date"],
                skipinitialspace=True,
            )
        elif method == "FromFile":
            seeding = pd.read_csv(
                self.seeding_config["seeding_file"].get(),
                converters={"subpop": lambda x: str(x)},
                parse_dates=["date"],
                skipinitialspace=True,
            )
        elif method == "NoSeeding":
            seeding = pd.DataFrame(columns=["date", "subpop"])
            return _DataFrame2NumbaDict(df=seeding, amounts=[], setup=setup)
        else:
            raise NotImplementedError(f"unknown seeding method [got: {method}]")

        # Sorting by date is very important here for the seeding format necessary !!!!
        seeding = seeding.sort_values(by="date", axis="index").reset_index()

        amounts = np.zeros(len(seeding))
        if method == "PoissonDistributed":
            amounts = np.random.poisson(seeding["amount"])
        elif method == "NegativeBinomialDistributed":
            raise ValueError("Seeding method 'NegativeBinomialDistributed' is not supported by flepiMoP anymore.")
            amounts = np.random.negative_binomial(n=5, p=5 / (seeding["amount"] + 5))
        elif method == "FolderDraw" or method == "FromFile":
            amounts = seeding["amount"]

        return _DataFrame2NumbaDict(df=seeding, amounts=amounts, setup=setup)

    def load_seeding(self, sim_id: int, setup) -> nb.typed.Dict:
        method = "NoSeeding"
        if "method" in self.seeding_config.keys():
            method = self.seeding_config["method"].as_str()
        if method not in ["FolderDraw", "SetInitialConditions", "InitialConditionsFolderDraw", "NoSeeding", "FromFile"]:
            raise NotImplementedError(
                f"Seeding method in inference run must be FolderDraw, SetInitialConditions, FromFile or InitialConditionsFolderDraw [got: {method}]"
            )
        return self.draw_seeding(sim_id=sim_id, setup=setup)

    def load_ic(self, sim_id: int, setup) -> nb.typed.Dict:
        return self.draw_ic(sim_id=sim_id, setup=setup)

    # Write seeding used to file
    def seeding_write(self, seeding, fname, extension):
        raise NotImplementedError(f"It is not yet possible to write the seeding to a file")
