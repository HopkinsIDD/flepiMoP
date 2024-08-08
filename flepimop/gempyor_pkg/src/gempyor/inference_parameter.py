import xarray as xr
import pandas as pd
import numpy as np
import confuse
from . import NPI


# TODO cast uper and lower bound as arrays
class InferenceParameters:
    """
    A class to manage inference parameters, in a vectorized way

    Parameters:
        global_config (confuse.ConfigView): The global configuration.
        subpop_names (list): The subpopulation names, in the right order
    """

    def __init__(self, global_config, subpop_names):
        self.ptypes = []
        self.pnames = []
        self.subpops = []
        self.pdists = []
        self.ubs = []
        self.lbs = []
        self.build_from_config(global_config, subpop_names)

    def add_modifier(self, pname, ptype, parameter_config, subpops):
        """
        Adds a modifier parameter to the parameters list.

        Args:
            pname (str): The parameter name.
            ptype (str): The parameter type.
            parameter_config (confuse.ConfigView): The configuration for the parameter.
            subpops (list): List of subpopulations affected by the modifier.
        """
        # identify spatial group
        affected_subpops = set(subpops)
        if (
            parameter_config["subpop"].exists()
            and parameter_config["subpop"].get() != "all"
        ):
            affected_subpops = {str(n.get()) for n in parameter_config["subpop"]}
        spatial_groups = NPI.helpers.get_spatial_groups(
            parameter_config, list(affected_subpops)
        )

        # ungrouped subpop (all affected subpop by default) have one parameter per subpop
        if spatial_groups["ungrouped"]:
            for sp in spatial_groups["ungrouped"]:
                self.add_single_parameter(
                    ptype=ptype,
                    pname=pname,
                    subpop=sp,
                    pdist=parameter_config["value"].as_random_distribution(),
                    lb=parameter_config["value"]["a"].get(),
                    ub=parameter_config["value"]["b"].get(),
                )

        # grouped subpop have one parameter per group
        if spatial_groups["grouped"]:
            for group in spatial_groups["grouped"]:
                self.add_single_parameter(
                    ptype=ptype,
                    pname=pname,
                    subpop=",".join(group),
                    pdist=parameter_config["value"].as_random_distribution(),
                    lb=parameter_config["value"]["a"].get(),
                    ub=parameter_config["value"]["b"].get(),
                )

    def add_single_parameter(self, ptype, pname, subpop, pdist, lb, ub):
        """
        Adds a single parameter to the parameters list.

        Args:
            ptype (str): The parameter type.
            pname (str): The parameter name.
            subpop (str): The subpopulation affected by the parameter.
            pdist: The distribution of the parameter.
            lb: The lower bound of the parameter.
            ub: The upper bound of the parameter.
        """
        self.ptypes.append(ptype)
        self.pnames.append(pname)
        self.subpops.append(subpop)
        self.pdists.append(pdist)
        self.ubs.append(ub)
        self.lbs.append(lb)

    def build_from_config(self, global_config, subpop_names):
        for config_part in ["seir_modifiers", "outcome_modifiers"]:
            if global_config[config_part].exists():
                for npi in global_config[config_part]["modifiers"].get():
                    if global_config[config_part]["modifiers"][npi][
                        "perturbation"
                    ].exists():
                        self.add_modifier(
                            pname=npi,
                            ptype=config_part,
                            parameter_config=global_config[config_part]["modifiers"][
                                npi
                            ],
                            subpops=subpop_names,
                        )

    def print_summary(self):
        print(f"There are {len(self.pnames)} parameters in the configuration.")
        for p_idx in range(self.get_dim()):
            print(
                f"{self.ptypes[p_idx]}::{self.pnames[p_idx]} in [{self.lbs[p_idx]}, {self.ubs[p_idx]}]"
                f"   >> affected subpop: {self.subpops[p_idx]}"
            )

    def __str__(self) -> str:
        from collections import Counter

        this_str = f"InferenceParameters: with {self.get_dim()} parameters: \n"
        for key, value in Counter(self.ptypes).items():
            this_str += f"    {key}: {value} parameters\n"

        return this_str

    def get_dim(self):
        return len(self.pnames)

    def get_parameters_for_subpop(self, subpop: str) -> list:
        """Returns the index parameters for a given subpopulation"""
        parameters = []
        for i, sp in enumerate(self.subpops):
            if sp == subpop:
                parameters.append(i)
        return parameters

    def __len__(self):
        """
        so one can use the built-in python len function
        """
        return len(self.pnames)

    def draw_initial(self, n_draw=1):
        """
        Draws initial parameter values.

        Args:
            n_draw (int): Number of draws, e.g the number of slots or walkers

        Returns:
            np.ndarray: Array of initial parameter values.
        """
        p0 = np.zeros((n_draw, self.get_dim()))
        for p_idx in range(self.get_dim()):
            p0[:, p_idx] = self.pdists[p_idx](n_draw)

        return p0

    # TODO: write a more granular method the return for a single parameter and correct the proposal like we did
    def check_in_bound(self, proposal) -> bool:
        """
        Checks if the proposal is within parameter bounds.

        Args:
            proposal: The proposed parameter values.

        Returns:
            bool: True if the proposal is within bounds, False otherwise.
        """
        if (
            self.hit_lbs(proposal=proposal).any()
            or self.hit_ubs(proposal=proposal).any()
        ):
            return False
        return True

    def hit_lbs(self, proposal) -> np.ndarray:
        return np.array((proposal < self.lbs))

    def hit_ubs(self, proposal) -> np.ndarray:
        """
        boolean vector of True if the parameter is bigger than the upper bound and False if not
        """
        return np.array((proposal > self.ubs))

    def inject_proposal(
        self,
        proposal,
        snpi_df=None,
        hnpi_df=None,
    ):
        """
        Injects the proposal into model inputs, at the right place.

        Args:
            proposal: The proposed parameter values.
            hnpi_df (pd.DataFrame): DataFrame for hnpi.
            snpi_df (pd.DataFrame): DataFrame for snpi.

        Returns:
            pd.DataFrame, pd.DataFrame: Modified hnpi_df and snpi_df.
        """
        snpi_df_mod = snpi_df.copy(deep=True)
        hnpi_df_mod = hnpi_df.copy(deep=True)

        # Ideally this should lie in each submodules, e.g NPI.inject, parameter.inject

        for p_idx in range(self.get_dim()):
            if self.ptypes[p_idx] == "seir_modifiers":
                snpi_df_mod.loc[
                    (snpi_df_mod["modifier_name"] == self.pnames[p_idx])
                    & (snpi_df_mod["subpop"] == self.subpops[p_idx]),
                    "value",
                ] = proposal[p_idx]
            elif self.ptypes[p_idx] == "outcome_modifiers":
                hnpi_df_mod.loc[
                    (hnpi_df_mod["modifier_name"] == self.pnames[p_idx])
                    & (hnpi_df_mod["subpop"] == self.subpops[p_idx]),
                    "value",
                ] = proposal[p_idx]
        return snpi_df_mod, hnpi_df_mod
