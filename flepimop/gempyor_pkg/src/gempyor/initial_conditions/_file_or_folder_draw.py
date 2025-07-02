"""Create initial conditions from a file or folder draw."""

__all__: tuple[str, ...] = ()


from datetime import date
from pathlib import Path
from typing import Final, Literal
import warnings

import numpy as np
import numpy.typing as npt
import pandas as pd
from pydantic import model_validator

from ..compartments import Compartments
from ..subpopulation_structure import SubpopulationStructure
from ..utils import read_df
from ..warnings import ConfigurationWarning
from ._base import InitialConditionsABC
from ._plugins import register_initial_conditions_plugin


_SEIR_OUTPUT_REQUIRED_COLUMNS: Final[set[str]] = {"date", "mc_value_type"}


def _read_initial_condition_from_tidydataframe(
    ic_df: pd.DataFrame,
    compartments: pd.DataFrame,
    subpopulation_structure: SubpopulationStructure,
    allow_missing_subpops: bool,
    allow_missing_compartments: bool,
    proportional_ic: bool,
) -> npt.NDArray[np.float64]:
    """
    Read the initial conditions from a tidy formatted DataFrame.

    Args:
        ic_df: The DataFrame containing the initial conditions.
        compartments: The compartments DataFrame, which is the `compartments` attribute
            of the `gempyor.compartments.Compartments` class.
        subpopulation_structure: The subpopulation structure object.
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
    y0 = np.zeros((len(compartments), subpopulation_structure.nsubpops))
    for pl_idx, pl in enumerate(subpopulation_structure.subpop_names):  #
        if pl in list(ic_df["subpop"]):
            states_pl = ic_df[ic_df["subpop"] == pl]
            for comp_idx, comp_name in compartments["name"].items():
                if "mc_name" in states_pl.columns:
                    ic_df_compartment_val = states_pl[states_pl["mc_name"] == comp_name][
                        "amount"
                    ]
                else:
                    filters = compartments.iloc[comp_idx].drop("name")
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
            total = subpopulation_structure.subpop_pop[pl_idx]
            if proportional_ic:
                total = 1.0
            y0[comp_idx, pl_idx] = total - y0[:, pl_idx].sum()
    if proportional_ic:
        y0 = y0 * subpopulation_structure.subpop_pop
    return y0


def _read_initial_condition_from_seir_output(
    seir_output: pd.DataFrame,
    compartments: pd.DataFrame,
    subpopulation_structure: SubpopulationStructure,
    start_date: date,
    allow_missing_subpops: bool,
    allow_missing_compartments: bool,
) -> np.ndarray:
    """
    Read the initial conditions from the SEIR output.

    Args:
        seir_output: The dataframe containing the seir output to build initial
            conditions from.
        compartments: The compartments DataFrame, which is the `compartments` attribute
            of the `gempyor.compartments.Compartments` class.
        subpopulation_structure: The subpopulation structure object.
        start_date: The start date for the simulation, determines the date to filter
            the given DataFrame from.
        allow_missing_subpops: Flag indicating whether missing subpopulations are
            allowed.
        allow_missing_compartments: Flag indicating whether missing compartments are
            allowed.

    Returns:
        The initial conditions array.

    Raises:
        ValueError: If there are required columns missing from the SEIR output.
        ValueError: If there are no entries in the SEIR output with the date
            `start_date` and 'mc_value_type' of 'prevalence'.
        ValueError: If there are more than one match for the compartments found
            in the SEIR output.
        ValueError: If there are not matches found for a given set of compartment
            filters in the SEIR output and `allow_missing_compartments` is `False`.
        NotImplementedError: If `allow_missing_subpops` is `True` and there are
            missing subpopulations in the SEIR output.

    """
    # Validation
    required_columns = _SEIR_OUTPUT_REQUIRED_COLUMNS | (
        {} if allow_missing_subpops else set(subpopulation_structure.subpop_names)
    )
    if missing_columns := required_columns - (columns := set(seir_output.columns.tolist())):
        raise ValueError(
            "The SEIR output initial conditions are missing required columns "
            f"{', '.join(sorted(missing_columns))}. Instead its columns "
            f"are {', '.join(sorted(columns))}."
        )
    # Wrangling
    start_date = start_date.strftime("%Y-%m-%d")
    seir_output["date"] = pd.Series(
        data=pd.to_datetime(seir_output["date"], utc=True), dtype="datetime64[ns, UTC]"
    ).dt.strftime("%Y-%m-%d")
    seir_output = seir_output[
        (seir_output["date"] == start_date) & (seir_output["mc_value_type"] == "prevalence")
    ]
    if seir_output.empty:
        raise ValueError(
            "No entries were found in the SEIR output initial conditions with "
            f"a date '{start_date}' and an MC value type of 'prevalence'."
        )
    # Loop over compartments & subpopulations
    y0 = np.zeros((len(compartments), subpopulation_structure.nsubpops))
    for row in compartments.itertuples():
        # Extract out a subset for this compartment & validate
        query = " & ".join(
            f"mc_{k}=='{v}'"
            for k, v in zip(["Index"] + compartments.columns.tolist(), row)
            if k not in {"Index", "name"}
        )
        initial_conditions_subset = seir_output.query(query)
        if (len_initial_conditions_subset := len(initial_conditions_subset)) > 1:
            raise ValueError(
                f"There were {len_initial_conditions_subset} matches found in "
                f"the SEIR output initial conditions for the filters {row}."
            )
        if initial_conditions_subset.empty:
            if not allow_missing_compartments:
                raise ValueError(
                    "There were no matches found in the SEIR output initial "
                    f"conditions for the filters {row}. Please specify or set "
                    "`allow_missing_compartments=True` in the configuration."
                )
            initial_conditions_subset = pd.DataFrame(
                data=0, columns=initial_conditions_subset.columns, index=[0]
            )
        elif (compartment_name := initial_conditions_subset["mc_name"].item()) != row.name:
            warnings.warn(
                "The SEIR output initial conditions compartment name, "
                f"'{compartment_name}', does not match the filtered "
                f"compartment MC name '{row.name}'.",
                RuntimeWarning,
            )
        # Loop over the subpopulations for this compartment
        for subpopulation_idx, subpopulation_name in enumerate(
            subpopulation_structure.subpop_names
        ):
            if subpopulation_name in columns:
                y0[row.Index, subpopulation_idx] = float(
                    initial_conditions_subset[subpopulation_name].item()
                )
            if allow_missing_subpops:
                raise NotImplementedError(
                    "Allowing missing subpopulations with SEIR output initial "
                    "conditions is currently not supported. If this is needed "
                    "for your use case please create a feature request."
                )
    return y0


class FileOrFolderDrawInitialConditions(InitialConditionsABC):
    """
    Initial conditions implementation that reads from a file.

    This class reads initial conditions from a specified file and returns them as a
    numpy array.
    """

    method: Literal[
        "SetInitialConditions",
        "SetInitialConditionsFolderDraw",
        "FromFile",
        "InitialConditionsFolderDraw",
    ]
    initial_file_type: str | None = None
    initial_conditions_file: Path | None = None
    ignore_population_checks: bool = False
    allow_missing_subpops: bool = False
    allow_missing_compartments: bool = False
    proportional_ic: bool = False

    @model_validator(mode="after")
    def _validate_attributes_required_by_method(
        self,
    ) -> "FileOrFolderDrawInitialConditions":
        """
        Validate that the attributes required by the `method` are set.

        Raises:
            ValueError: If `meta` is not set.
            ValueError: When `method` is 'SetInitialConditionsFolderDraw' or
                'InitialConditionsFolderDraw' and `initial_file_type` is not set.
            ValueError: When `method` is 'SetInitialConditions' or 'FromFile' and
                `initial_conditions_file` is not set.
            ValueError: When `method` is 'FromFile' or 'InitialConditionsFolderDraw'
                and `time_setup` is not set.
        """
        if self.meta is None:
            raise ValueError(
                "The `meta` attribute must be set when using one of the following "
                "methods: 'SetInitialConditions', 'SetInitialConditionsFolderDraw', "
                "'FromFile', 'InitialConditionsFolderDraw'."
            )
        if self.method in {"SetInitialConditionsFolderDraw", "InitialConditionsFolderDraw"}:
            if self.initial_file_type is None:
                raise ValueError(
                    "The `initial_file_type` attribute must be set when using "
                    "'SetInitialConditionsFolderDraw' or 'InitialConditionsFolderDraw'."
                )
            if self.initial_conditions_file is not None:
                warnings.warn(
                    "The `initial_conditions_file` attribute as been intentionally set "
                    "to a non-default value but is not used when initial conditions "
                    f"method is {self.method}.",
                    ConfigurationWarning,
                )
        if self.method in {"SetInitialConditions", "FromFile"}:
            if self.initial_conditions_file is None:
                raise ValueError(
                    "The `initial_conditions_file` attribute must be set when using "
                    "'SetInitialConditions' or 'FromFile'."
                )
            if self.initial_file_type is not None:
                warnings.warn(
                    "The `initial_file_type` attribute as been intentionally set "
                    "to a non-default value but is not used when initial conditions "
                    f"method is {self.method}.",
                    ConfigurationWarning,
                )
        if self.method in {"FromFile", "InitialConditionsFolderDraw"}:
            if self.time_setup is None:
                raise ValueError(
                    "The `time_setup` attribute must be set when using "
                    "'FromFile' or 'InitialConditionsFolderDraw'."
                )
            if self.proportional_ic is True:
                warnings.warn(
                    "The `proportional_ic` attribute as been intentionally set to a "
                    "non-default value but is not used when initial conditions method "
                    f"is {self.method}.",
                    ConfigurationWarning,
                )
        return self

    def create_initial_conditions(
        self,
        sim_id: int,
        compartments: Compartments,
        subpopulation_structure: SubpopulationStructure,
    ) -> npt.NDArray[np.float64]:
        """
        Produce an array of initial conditions by reading from a file.

        Args:
            sim_id: The simulation ID.
            compartments: The compartments object.
            subpopulation_structure: The subpopulation structure object.

        Returns:
            A numpy array of initial conditions for the simulation.
        """
        if self.method in {"SetInitialConditionsFolderDraw", "InitialConditionsFolderDraw"}:
            initial_conditions = self.meta.read_sim_id(self.initial_file_type, sim_id)
        else:
            initial_conditions = read_df(
                self.meta.path_prefix / self.initial_conditions_file
            )
        if self.method in {"SetInitialConditions", "SetInitialConditionsFolderDraw"}:
            return _read_initial_condition_from_tidydataframe(
                initial_conditions,
                compartments.compartments,
                subpopulation_structure,
                self.allow_missing_subpops,
                self.allow_missing_compartments,
                self.proportional_ic,
            )
        return _read_initial_condition_from_seir_output(
            initial_conditions,
            compartments.compartments,
            subpopulation_structure,
            self.time_setup.start_date,
            self.allow_missing_subpops,
            self.allow_missing_compartments,
        )


register_initial_conditions_plugin(FileOrFolderDrawInitialConditions)
