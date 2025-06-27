"""The core representation of initial conditions in `gempyor`."""

__all__: tuple[str, ...] = ()


import confuse
import numpy as np

from ..utils import read_df
from ._utils import check_population
from ._readers import (
    read_initial_condition_from_seir_output,
    read_initial_condition_from_tidydataframe,
)


class InitialConditions:
    """Represents the initial conditions for a simulation."""

    def __init__(
        self,
        config: confuse.ConfigView,
        path_prefix: str = ".",
    ):
        """
        Initialize an initial conditions object from configuration.

        Args:
            config: A confuse configuration object containing initial conditions
                settings.
            path_prefix: The prefix path for file paths in the configuration.
        """
        self.initial_conditions_config = config
        self.path_prefix = path_prefix

        # default values, overwritten below
        self.ignore_population_checks = False
        self.allow_missing_subpops = False
        self.allow_missing_compartments = False
        self.proportional_ic = False

        if self.initial_conditions_config is not None:
            if "ignore_population_checks" in self.initial_conditions_config.keys():
                self.ignore_population_checks = self.initial_conditions_config[
                    "ignore_population_checks"
                ].get(bool)
            if "allow_missing_subpops" in self.initial_conditions_config.keys():
                self.allow_missing_subpops = self.initial_conditions_config[
                    "allow_missing_subpops"
                ].get(bool)
            if "allow_missing_compartments" in self.initial_conditions_config.keys():
                self.allow_missing_compartments = self.initial_conditions_config[
                    "allow_missing_compartments"
                ].get(bool)
            if "proportional" in self.initial_conditions_config.keys():
                self.proportional_ic = self.initial_conditions_config["proportional"].get(
                    bool
                )

    def get_from_config(self, sim_id: int, modinf) -> np.ndarray:
        """
        Produce an array of initial conditions from the configuration.

        Args:
            sim_id: The simulation ID.
            modinf: The model information object.

        Returns:
            A numpy array of initial conditions for the simulation.
        """
        method = "Default"
        if (
            self.initial_conditions_config is not None
            and "method" in self.initial_conditions_config.keys()
        ):
            method = self.initial_conditions_config["method"].as_str()

        if method == "Default":
            ## JK : This could be specified in the config
            y0 = np.zeros((modinf.compartments.compartments.shape[0], modinf.nsubpops))
            y0[0, :] = modinf.subpop_pop
            return y0  # we finish here: no rest and not proportionality applies

        if method in {"SetInitialConditions", "SetInitialConditionsFolderDraw"}:
            if method == "SetInitialConditionsFolderDraw":
                ic_df = modinf.read_simID(
                    ftype=self.initial_conditions_config["initial_file_type"], sim_id=sim_id
                )
            else:
                ic_df = read_df(
                    self.path_prefix
                    / self.initial_conditions_config["initial_conditions_file"].get(),
                )
            y0 = read_initial_condition_from_tidydataframe(
                ic_df=ic_df,
                modinf=modinf,
                allow_missing_compartments=self.allow_missing_compartments,
                allow_missing_subpops=self.allow_missing_subpops,
                proportional_ic=self.proportional_ic,
            )

        elif method in {"InitialConditionsFolderDraw", "FromFile"}:
            if method == "InitialConditionsFolderDraw":
                ic_df = modinf.read_simID(
                    ftype=self.initial_conditions_config["initial_file_type"].get(),
                    sim_id=sim_id,
                )
            elif method == "FromFile":
                ic_df = read_df(
                    self.path_prefix
                    / self.initial_conditions_config["initial_conditions_file"].get(),
                )

            y0 = read_initial_condition_from_seir_output(
                ic_df=ic_df,
                modinf=modinf,
                allow_missing_compartments=self.allow_missing_compartments,
                allow_missing_subpops=self.allow_missing_subpops,
            )
        else:
            raise NotImplementedError(
                f"Unknown initial conditions method [received: '{method}']."
            )

        # check that the inputted values sums to the subpop population:
        check_population(
            y0,
            modinf.subpop_struct.subpop_names,
            modinf.subpop_pop,
            ignore_population_checks=self.ignore_population_checks,
        )
        return y0

    def get_from_file(self, sim_id: int, modinf) -> np.ndarray:
        """
        Produce an array of initial conditions from the configuration.

        This method is a wrapper around `get_from_config` to maintain compatibility
        with existing code that expects this method. Direct usage of this method is
        deprecated in favor of `get_from_config`.

        Args:
            sim_id: The simulation ID.
            modinf: The model information object.

        Returns:
            A numpy array of initial conditions for the simulation.
        """
        return self.get_from_config(sim_id=sim_id, modinf=modinf)
