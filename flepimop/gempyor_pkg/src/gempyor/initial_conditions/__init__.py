"""Provides functionality for handling initial conditions."""

__all__ = (
    "InitialConditions",
    "check_population",
    "initial_conditions_factory",
    "read_initial_condition_from_seir_output",
    "read_initial_condition_from_tidydataframe",
)

import logging
import warnings

import confuse
import numpy as np
import numpy.typing as npt
import pandas as pd

from ..utils import read_df, search_and_import_plugins_class


logger = logging.getLogger(__name__)


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


def initial_conditions_factory(
    config: confuse.ConfigView, path_prefix: str = "."
) -> InitialConditions:
    """
    Create an initial conditions object from config or plugin.

    Args:
        config: The configuration object containing initial conditions settings.
        path_prefix: The prefix path for file paths in the configuration.

    Returns:
        An instance of the `InitialConditions` class created from configuration or a
        plugin subclass.
    """
    if config is not None and "method" in config.keys():
        if config["method"].as_str() == "plugin":
            klass = search_and_import_plugins_class(
                plugin_file_path=config["plugin_file_path"].as_str(),
                class_name="InitialConditions",
                config=config,
                path_prefix=path_prefix,
            )
            return klass
    return InitialConditions(config, path_prefix=path_prefix)


def check_population(
    y0: np.ndarray,
    subpop_names: list[str],
    subpop_pop: npt.NDArray[np.int64],
    ignore_population_checks: bool = False,
) -> None:
    """
    Check that the initial conditions match the population sizes in the model.

    Args:
        y0: The initial conditions array where the row dimension corresponds to the
            compartments and the column dimension corresponds to the subpopulations.
        subpop_names: A list of subpopulation names.
        subpop_pop: A numpy array containing the population sizes for each
            subpopulation.
        ignore_population_checks: If `True`, ignore population checks.

    Raises:
        ValueError: If the initial conditions do not match the population sizes and
            `ignore_population_checks` is `False`.
    """
    error = False
    for pl_idx, pl in enumerate(subpop_names):
        n_y0 = y0[:, pl_idx].sum()
        n_pop = subpop_pop[pl_idx]
        if abs(n_y0 - n_pop) > 1:
            error = True
            warnings.warn(
                f"`subpop_names` '{pl}' (idx: plx_idx) has a population from initial "
                f"condition of '{n_y0}' while population geodata is '{n_pop}'. "
                f"(absolute difference should be <1, here is '{abs(n_y0-n_pop)}')."
            )
    if error:
        if not ignore_population_checks:
            raise ValueError(
                "Geodata and initial condition do not agree on population size. "
                "Use `ignore_population_checks: True` to ignore."
            )
        warnings.warn(
            "Population mismatch errors ignored because `ignore_population_checks` is "
            "set to `True`. Execution will continue, but this is not recommended.",
            UserWarning,
        )


def read_initial_condition_from_tidydataframe(
    ic_df: pd.DataFrame,
    modinf,
    allow_missing_subpops: bool,
    allow_missing_compartments: bool,
    proportional_ic: bool = False,
):
    """
    Read the initial conditions from a tidy formatted DataFrame.

    Args:
        ic_df: The DataFrame containing the initial conditions.
        modinf: The model information object.
        allow_missing_subpops: Flag indicating whether missing subpopulations are
            allowed.
        allow_missing_compartments: Flag indicating whether missing compartments are
            allowed.
        proportional_ic: If `True`, the initial conditions will be set proportionally
            to the subpopulation sizes.

    Returns:
        The initial conditions array.

    Raises:
        ValueError: If the compartment filters in the initial conditions DataFrame
            are not unique.
        ValueError: If the compartments are not unique in the initial conditions
            DataFrame.
        RuntimeError: If `allow_missing_subpops` is `True`.
        ValueError: If a subpopulation does not exist in the initial conditions
            DataFrame and `allow_missing_subpops` is `False`.
    """
    rests = []  # Places to allocate the rest of the population
    y0 = np.zeros((modinf.compartments.compartments.shape[0], modinf.nsubpops))
    for pl_idx, pl in enumerate(modinf.subpop_struct.subpop_names):  #
        if pl in list(ic_df["subpop"]):
            states_pl = ic_df[ic_df["subpop"] == pl]
            for comp_idx, comp_name in modinf.compartments.compartments["name"].items():
                if "mc_name" in states_pl.columns:
                    ic_df_compartment_val = states_pl[states_pl["mc_name"] == comp_name][
                        "amount"
                    ]
                else:
                    filters = modinf.compartments.compartments.iloc[comp_idx].drop("name")
                    ic_df_compartment_val = states_pl.copy()
                    for mc_name, mc_value in filters.items():
                        ic_df_compartment_val = ic_df_compartment_val[
                            ic_df_compartment_val["mc_" + mc_name] == mc_value
                        ]["amount"]
                if len(ic_df_compartment_val) > 1:
                    raise ValueError(
                        f"Several ('{len(ic_df_compartment_val)}') rows are matches "
                        f"for compartment '{comp_name}' in init file: filters "
                        f"returned '{ic_df_compartment_val}'"
                    )
                if ic_df_compartment_val.empty:
                    if allow_missing_compartments:
                        ic_df_compartment_val = 0.0
                    else:
                        raise ValueError(
                            f"Multiple rows match for compartment '{comp_name}' in the "
                            "initial conditions file; ensure each compartment has a "
                            f"unique entry. Filters used: '{filters.to_dict()}'. "
                            f"Matches: '{ic_df_compartment_val.tolist()}'."
                        )
                if "rest" in str(ic_df_compartment_val).strip().lower():
                    rests.append([comp_idx, pl_idx])
                else:
                    if isinstance(
                        ic_df_compartment_val, pd.Series
                    ):  # it can also be float if we allow allow_missing_compartments
                        ic_df_compartment_val = float(ic_df_compartment_val.iloc[0])
                    y0[comp_idx, pl_idx] = float(ic_df_compartment_val)
        elif allow_missing_subpops:
            logger.critical(
                "No initial conditions for for subpop %s, assuming everyone"
                "(n=%u) in the first meta-compartment (%s)",
                pl,
                modinf.subpop_pop[pl_idx],
                modinf.compartments.compartments["name"].iloc[0],
            )
            raise RuntimeError(
                "There is a bug; report this message. Past implementation was buggy."
            )
        else:
            raise ValueError(
                f"Subpop '{pl}' does not exist in `initial_conditions::states_file`. "
                f"You can set `allow_missing_subpops=TRUE` to bypass this error."
            )
    if rests:  # not empty
        for comp_idx, pl_idx in rests:
            total = modinf.subpop_pop[pl_idx]
            if proportional_ic:
                total = 1.0
            y0[comp_idx, pl_idx] = total - y0[:, pl_idx].sum()

    if proportional_ic:
        y0 = y0 * modinf.subpop_pop
    return y0


def read_initial_condition_from_seir_output(
    ic_df: pd.DataFrame,
    modinf,
    allow_missing_subpops: bool,
    allow_missing_compartments: bool,
) -> np.ndarray:
    """
    Read the initial conditions from the SEIR output.

    Args:
        ic_df: The dataframe containing the initial conditions.
        modinf: The model information object.
        allow_missing_subpops: Flag indicating whether missing subpopulations are
            allowed.
        allow_missing_compartments: Flag indicating whether missing compartments are
            allowed.

    Returns:
        The initial conditions array.

    Raises:
        ValueError: If there is no entry for the initial time ti in the provided
            initial_conditions::states_file.
        ValueError: If there are multiple rows matching the compartment in the init
            file.
        ValueError: If the compartment cannot be set in the subpopulation.
        ValueError: If the subpopulation does not exist in
            initial_conditions::states_file.

    """
    # annoying conversion because sometime the parquet columns get attributed a timezone...
    ic_df["date"] = pd.to_datetime(ic_df["date"], utc=True)  # force date to be UTC
    ic_df["date"] = ic_df["date"].dt.date
    ic_df["date"] = ic_df["date"].astype(str)

    ic_df = ic_df[
        (ic_df["date"] == str(modinf.ti)) & (ic_df["mc_value_type"] == "prevalence")
    ]
    if ic_df.empty:
        raise ValueError(
            f"No entry provided for initial time `ti` in the "
            f"`initial_conditions::states_file.` `ti`: '{modinf.ti}'."
        )
    y0 = np.zeros((modinf.compartments.compartments.shape[0], modinf.nsubpops))

    for comp_idx, comp_name in modinf.compartments.compartments["name"].items():
        # rely on all the mc's instead of mc_name to avoid errors due to e.g order.
        # before: only
        # ic_df_compartment = ic_df[ic_df["mc_name"] == comp_name]
        filters = modinf.compartments.compartments.iloc[comp_idx].drop("name")
        ic_df_compartment = ic_df.copy()
        for mc_name, mc_value in filters.items():
            ic_df_compartment = ic_df_compartment[
                ic_df_compartment["mc_" + mc_name] == mc_value
            ]

        if len(ic_df_compartment) > 1:
            # ic_df_compartment = ic_df_compartment.iloc[0]
            raise ValueError(
                f"Several ('{len(ic_df_compartment)}') rows are matches for "
                f"compartment '{comp_name}' in init file: filter '{filters}'. "
                f"returned: '{ic_df_compartment}'."
            )
        if ic_df_compartment.empty:
            if not allow_missing_compartments:
                raise ValueError(
                    f"Initial Conditions: could not set compartment '{comp_name}' "
                    f"(id: '{comp_idx}') in subpop '{pl}' (id: '{pl_idx}'). The data "
                    f"from the init file is '{ic_df_compartment[pl]}'."
                )
            ic_df_compartment = pd.DataFrame(
                0, columns=ic_df_compartment.columns, index=[0]
            )
        elif ic_df_compartment["mc_name"].iloc[0] != comp_name:
            warnings.warn(
                f"{ic_df_compartment['mc_name'].iloc[0]} does not match "
                f"compartment `mc_name` {comp_name}."
            )

        for pl_idx, pl in enumerate(modinf.subpop_struct.subpop_names):
            if pl in ic_df.columns:
                y0[comp_idx, pl_idx] = float(ic_df_compartment[pl].iloc[0])
            elif allow_missing_subpops:
                raise RuntimeError(
                    "There is a bug; report this message. Past implementation was buggy"
                )
            else:
                raise ValueError(
                    f"Subpop '{pl}' does not exist in `initial_conditions::states_file`. "
                    f"You can set `allow_missing_subpops=TRUE` to bypass this error."
                )
    return y0
