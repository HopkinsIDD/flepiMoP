"""
Managing inference parameters in a vectorized way.
"""

__all__ = ("InferenceParameters",)


from collections import Counter
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
from confuse import ConfigView

from . import NPI
from .distributions import DistributionABC, distribution_from_confuse_config


class InferenceParameters:
    """
    A class to manage vectorized inference parameters.
    """

    def __init__(self, global_config: ConfigView, subpop_names: list[str]) -> None:
        """
        Initializes the `InferenceParameters` instance.

        This constructor sets up the inference parameters based on the provided global
        configuration and subpopulation names by calling the `build_from_config` method.

        Args:
            global_config: The global configuration represented as a
                `confuse.ConfigView`.
            subpop_names: A list of subpopulation names to be used in
                the configuration.
        """
        self.ptypes: list[Literal["outcome_modifiers", "seir_modifiers"]] = []
        self.pnames: list[str] = []
        self.subpops: list[str] = []
        self.pdists: list[DistributionABC] = []
        self.build_from_config(global_config, subpop_names)

    def __str__(self) -> str:
        return f"InferenceParameters: with {len(self)} parameters: \n" + "\n".join(
            f"    {key}: {value} parameters" for key, value in Counter(self.ptypes).items()
        )

    def __len__(self) -> int:
        return len(self.pnames)

    def print_summary(self) -> None:
        """
        Prints a summary of the inference parameters.

        Produces a summary of the parameters, including their types, names, bounds and
        relevant subpopulations. This is useful for debugging and understanding the
        configuration.
        """
        dim = len(self)
        print(f"There are {dim} parameters in the configuration.")
        for p_idx in range(dim):
            lower, upper = self.pdists[p_idx].support
            print(
                f"{self.ptypes[p_idx]}::{self.pnames[p_idx]} in "
                f"[{lower}, {upper}] >> affected subpop: {self.subpops[p_idx]}"
            )

    def build_from_config(self, global_config: ConfigView, subpop_names: list[str]) -> None:
        """
        Constructs the inference parameters from the global configuration.

        Args:
            global_config: The global configuration represented as a
                `confuse.ConfigView`.
            subpop_names: A list of subpopulation names to be used in
                the configuration.
        """
        for config_part in ["seir_modifiers", "outcome_modifiers"]:
            if global_config[config_part].exists():
                for npi in global_config[config_part]["modifiers"].get():
                    if global_config[config_part]["modifiers"][npi][
                        "perturbation"
                    ].exists():
                        self.add_modifier(
                            pname=npi,
                            ptype=config_part,
                            parameter_config=global_config[config_part]["modifiers"][npi],
                            subpops=subpop_names,
                        )

    def add_modifier(
        self,
        pname: str,
        ptype: Literal["outcome_modifiers", "seir_modifiers"],
        parameter_config: ConfigView,
        subpops: list[str],
    ) -> None:
        """
        Adds a modifier parameter to the inference parameters representation.

        Args:
            pname: The name of the parameter.
            ptype: The parameter type, must be one of "outcome_modifiers" or
                "seir_modifiers".
            parameter_config: The confuse representation of the parameter configuration.
            subpops: A list of subpopulations affected by the modifier.
        """
        # identify spatial group
        affected_subpops = set(subpops)

        if parameter_config["method"].get() == "SinglePeriodModifier":
            if (
                parameter_config["subpop"].exists()
                and parameter_config["subpop"].get() != "all"
            ):
                affected_subpops = {str(n.get()) for n in parameter_config["subpop"]}
            spatial_groups = NPI.helpers.get_spatial_groups(
                parameter_config, list(affected_subpops)
            )
            # ungrouped subpop (all affected subpop by
            # default) have one parameter per subpop
            if spatial_groups["ungrouped"]:
                for sp in spatial_groups["ungrouped"]:
                    dist = distribution_from_confuse_config(parameter_config["value"])
                    self.add_single_parameter(
                        ptype=ptype,
                        pname=pname,
                        subpop=sp,
                        pdist=dist,
                    )
            # grouped subpop have one parameter per group
            if spatial_groups["grouped"]:
                for group in spatial_groups["grouped"]:
                    dist = distribution_from_confuse_config(parameter_config["value"])
                    self.add_single_parameter(
                        ptype=ptype,
                        pname=pname,
                        subpop=",".join(group),
                        pdist=dist,
                    )
        elif parameter_config["method"].get() == "MultiPeriodModifier":
            affected_subpops_grp = []
            for grp_config in parameter_config["groups"]:
                if grp_config["subpop"].get() == "all":
                    affected_subpops_grp = affected_subpops
                else:
                    affected_subpops_grp += [str(n.get()) for n in grp_config["subpop"]]
            affected_subpops = list(set(affected_subpops_grp))
            spatial_groups = []
            for grp_config in parameter_config["groups"]:
                if grp_config["subpop"].get() == "all":
                    affected_subpops_grp = affected_subpops
                else:
                    affected_subpops_grp = [str(n.get()) for n in grp_config["subpop"]]

                this_spatial_group = NPI.helpers.get_spatial_groups(
                    grp_config, affected_subpops_grp
                )
                # ungrouped subpop (all affected subpop by
                # default) have one parameter per subpop
                if this_spatial_group["ungrouped"]:
                    for sp in this_spatial_group["ungrouped"]:
                        dist = distribution_from_confuse_config(parameter_config["value"])
                        self.add_single_parameter(
                            ptype=ptype,
                            pname=pname,
                            subpop=sp,
                            pdist=dist,
                        )
                # grouped subpop have one parameter per group
                if this_spatial_group["grouped"]:
                    for group in this_spatial_group["grouped"]:
                        dist = distribution_from_confuse_config(parameter_config["value"])
                        self.add_single_parameter(
                            ptype=ptype,
                            pname=pname,
                            subpop=",".join(group),
                            pdist=dist,
                        )
        else:
            raise ValueError(f"Unknown method {parameter_config['method']}")

    def add_single_parameter(
        self,
        ptype: Literal["outcome_modifiers", "seir_modifiers"],
        pname: str,
        subpop: str,
        pdist: DistributionABC,
    ) -> None:
        """
        Adds a single parameter to the inference parameters representation.

        Args:
            ptype: The parameter type, must be one of "outcome_modifiers" or
                "seir_modifiers".
            pname: The parameter name.
            subpop: The subpopulation affected by the parameter.
            pdist: The distribution of the parameter.
        """
        self.ptypes.append(ptype)
        self.pnames.append(pname)
        self.subpops.append(subpop)
        self.pdists.append(pdist)

    def get_dim(self) -> int:
        """
        Get the dimension of the parameter space.

        Returns:
            The dimension of the parameter space, which is a non-negative integer.
        """
        return len(self)

    def get_parameters_for_subpop(self, subpop: str) -> list[int]:
        """
        Get the indices of parameters relevant for a specific subpopulation.

        Args:
            subpop: The name of the subpopulation to pull parameters indexes for.

        Returns:
            A list of indices corresponding to parameters that affect the specified
            subpopulation.
        """
        return [i for i, s in enumerate(self.subpops) if s == subpop]

    def draw_initial(self, n_draw: int = 1) -> npt.NDArray[np.float64]:
        """
        Draws initial parameter values.

        Args:
            n_draw: Number of draws, e.g the number of slots or walkers.

        Returns:
            Array of initial parameter values with shape (`n_draw`, dim).
        """
        dim = len(self)
        p0 = np.zeros((n_draw, dim))
        for p_idx in range(dim):
            p0[:, p_idx] = self.pdists[p_idx].sample(size=n_draw)
        return p0

    def check_in_bound(self, proposal: npt.NDArray[np.float64]) -> bool:
        """
        Checks if the proposal is within parameter bounds.

        Args:
            proposal: The proposed parameter values.

        Returns:
            `True` if the proposal is within bounds, `False` otherwise.
        """
        lower_bounds, upper_bounds = zip(*[dist.support for dist in self.pdists])
        return np.logical_and(
            np.greater_equal(proposal, lower_bounds),
            np.less_equal(proposal, upper_bounds),
        ).all()

    def inject_proposal(
        self,
        proposal: npt.NDArray[np.float64],
        snpi_df: pd.DataFrame,
        hnpi_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Injects the proposal into model inputs, at the right place.

        Args:
            proposal: The proposed parameter values.
            hnpi_df (pd.DataFrame): DataFrame for hnpi.
            snpi_df (pd.DataFrame): DataFrame for snpi.

        Returns:
            A tuple of modified DataFrames, the first one for SEIR modifiers and the
            second for outcome modifiers.
        """
        snpi_df_mod = snpi_df.copy(deep=True)
        hnpi_df_mod = hnpi_df.copy(deep=True)
        for p_idx in range(len(self)):
            df = snpi_df_mod if self.ptypes[p_idx] == "seir_modifiers" else hnpi_df_mod
            df.loc[
                (df["modifier_name"] == self.pnames[p_idx])
                & (df["subpop"] == self.subpops[p_idx]),
                "value",
            ] = proposal[p_idx]
        return snpi_df_mod, hnpi_df_mod
