"""Read initial conditions from various formats."""

__all__: tuple[str, ...] = ()


import logging
import warnings

import pandas as pd
import numpy as np


logger = logging.getLogger(__name__)


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
