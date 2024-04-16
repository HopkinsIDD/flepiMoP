from typing import Dict

import numpy as np
import pandas as pd
from numba.typed import Dict
import confuse
import logging
from .simulation_component import SimulationComponent
from . import utils
from .utils import read_df

logger = logging.getLogger(__name__)


class InitialConditions(SimulationComponent):
    def __init__(self, config: confuse.ConfigView):
        self.initial_conditions_config = config

    def get_from_config(self, sim_id: int, setup) -> np.ndarray:
        method = "Default"
        if self.initial_conditions_config is not None and "method" in self.initial_conditions_config.keys():
            method = self.initial_conditions_config["method"].as_str()

        if method == "Default":
            ## JK : This could be specified in the config
            y0 = np.zeros((setup.compartments.compartments.shape[0], setup.nsubpops))
            y0[0, :] = setup.subpop_pop
            return y0  # we finish here: no rest and not proportionallity applies

        allow_missing_subpops = False
        allow_missing_compartments = False
        if "allow_missing_subpops" in self.initial_conditions_config.keys():
            if self.initial_conditions_config["allow_missing_subpops"].get():
                allow_missing_subpops = True
        if "allow_missing_compartments" in self.initial_conditions_config.keys():
            if self.initial_conditions_config["allow_missing_compartments"].get():
                allow_missing_compartments = True

        # Places to allocate the rest of the population
        rests = []

        if method == "SetInitialConditions" or method == "SetInitialConditionsFolderDraw":
            #  TODO Think about     - Does not support the new way of doing compartment indexing
            if method == "SetInitialConditionsFolderDraw":
                ic_df = setup.read_simID(ftype=self.initial_conditions_config["initial_file_type"], sim_id=sim_id)
            else:
                ic_df = read_df(
                    self.initial_conditions_config["initial_conditions_file"].get(),
                )

            y0 = np.zeros((setup.compartments.compartments.shape[0], setup.nsubpops))
            for pl_idx, pl in enumerate(setup.subpop_struct.subpop_names):  #
                if pl in list(ic_df["subpop"]):
                    states_pl = ic_df[ic_df["subpop"] == pl]
                    for comp_idx, comp_name in setup.compartments.compartments["name"].items():
                        if "mc_name" in states_pl.columns:
                            ic_df_compartment_val = states_pl[states_pl["mc_name"] == comp_name]["amount"]
                        else:
                            filters = setup.compartments.compartments.iloc[comp_idx].drop("name")
                            ic_df_compartment_val = states_pl.copy()
                            for mc_name, mc_value in filters.items():
                                ic_df_compartment_val = ic_df_compartment_val[ic_df_compartment_val["mc_" + mc_name] == mc_value][
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
                                    f"Initial Conditions: Could not set compartment {comp_name} (id: {comp_idx}) in subpop {pl} (id: {pl_idx}). The data from the init file is {states_pl}. \n \
                                                Use 'allow_missing_compartments' to default to 0 for compartments without initial conditions"
                                )
                        if "rest" in str(ic_df_compartment_val).strip().lower():
                            rests.append([comp_idx, pl_idx])
                        else:
                            if isinstance(ic_df_compartment_val, pd.Series): # it can also be float if we allow allow_missing_compartments
                                ic_df_compartment_val = float(ic_df_compartment_val.iloc[0])
                            y0[comp_idx, pl_idx] = float(ic_df_compartment_val)
                elif allow_missing_subpops:
                    logger.critical(
                        f"No initial conditions for for subpop {pl}, assuming everyone (n={setup.subpop_pop[pl_idx]}) in the first metacompartment ({setup.compartments.compartments['name'].iloc[0]})"
                    )
                    if "proportional" in self.initial_conditions_config.keys():
                        if self.initial_conditions_config["proportional"].get():
                            y0[0, pl_idx] = 1.0
                        else:
                            y0[0, pl_idx] = setup.subpop_pop[pl_idx]
                    else:
                        y0[0, pl_idx] = setup.subpop_pop[pl_idx]
                else:
                    raise ValueError(
                        f"subpop {pl} does not exist in initial_conditions::states_file. You can set allow_missing_subpops=TRUE to bypass this error"
                    )
        elif method == "InitialConditionsFolderDraw" or method == "FromFile":
            if method == "InitialConditionsFolderDraw":
                ic_df = setup.read_simID(ftype=self.initial_conditions_config["initial_file_type"].get(), sim_id=sim_id)
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
            y0 = np.zeros((setup.compartments.compartments.shape[0], setup.nsubpops))

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
                            f"Initial Conditions: Could not set compartment {comp_name} (id: {comp_idx}) in subpop {pl} (id: {pl_idx}). The data from the init file is {ic_df_compartment[pl]}."
                        )
                elif ic_df_compartment["mc_name"].iloc[0] != comp_name:
                    print(
                        f"WARNING: init file mc_name {ic_df_compartment['mc_name'].iloc[0]} does not match compartment mc_name {comp_name}"
                    )

                for pl_idx, pl in enumerate(setup.subpop_struct.subpop_names):
                    if pl in ic_df.columns:
                        y0[comp_idx, pl_idx] = float(ic_df_compartment[pl].iloc[0])
                    elif allow_missing_subpops:
                        logger.critical(
                            f"No initial conditions for for subpop {pl}, assuming everyone (n={setup.subpop_pop[pl_idx]}) in the first metacompartments ({setup.compartments.compartments['name'].iloc[0]})"
                        )
                        if "proportion" in self.initial_conditions_config.keys():
                            if self.initial_conditions_config["proportion"].get():
                                y0[0, pl_idx] = 1.0
                        y0[0, pl_idx] = setup.subpop_pop[pl_idx]
                    else:
                        raise ValueError(
                            f"subpop {pl} does not exist in initial_conditions::states_file. You can set allow_missing_subpops=TRUE to bypass this error"
                        )
        else:
            raise NotImplementedError(f"unknown initial conditions method [got: {method}]")

        # rest
        if rests:  # not empty
            for comp_idx, pl_idx in rests:
                total = setup.subpop_pop[pl_idx]
                if "proportional" in self.initial_conditions_config.keys():
                    if self.initial_conditions_config["proportional"].get():
                        total = 1.0
                y0[comp_idx, pl_idx] = total - y0[:, pl_idx].sum()

        if "proportional" in self.initial_conditions_config.keys():
            if self.initial_conditions_config["proportional"].get():
                y0 = y0 * setup.subpop_pop

        # check that the inputed values sums to the subpop population:
        error = False
        for pl_idx, pl in enumerate(setup.subpop_struct.subpop_names):
            n_y0 = y0[:, pl_idx].sum()
            n_pop = setup.subpop_pop[pl_idx]
            if abs(n_y0 - n_pop) > 1:
                error = True
                print(
                    f"ERROR: subpop_names {pl} (idx: pl_idx) has a population from initial condition of {n_y0} while population from geodata is {n_pop} (absolute difference should be < 1, here is {abs(n_y0-n_pop)})"
                )
        ignore_population_checks = False
        if "ignore_population_checks" in self.initial_conditions_config.keys():
            if self.initial_conditions_config["ignore_population_checks"].get():
                ignore_population_checks = True
        if error and not ignore_population_checks:
            raise ValueError(
                """ geodata and initial condition do not agree on population size (see messages above). Use ignore_population_checks: True to ignore"""
            )
        elif error and ignore_population_checks:
            print(
                """ Ignoring the previous population mismatch errors because you added flag 'ignore_population_checks'. This is dangerous"""
            )
        return y0

    def get_from_file(self, sim_id: int, setup) -> np.ndarray:
        return self.get_from_config(sim_id=sim_id, setup=setup)

# TODO: rename config to initial_conditions_config as it shadows the global config


def InitialConditionsFactory(config: confuse.ConfigView):
    if config is not None and "method" in config.keys():
            if config["method"].as_str() == "plugin":
                klass = utils.search_and_import_plugins_class(
                    plugin_file_path=config["plugin_file_path"].as_str(), 
                    class_name="InitialConditions",
                    config=config
                    )
                return klass
    return InitialConditions(config)
