"""
Abstractions for interacting with models as a monolithic object.

Defines the `ModelInfo` class (and associated methods), used for setting up and
managing the configuration of a simulation. The primary focuses of a `ModelInfo` object
are parsing and validating config details (including time frames, subpop info,
model parameters, outcomes) and file handling for input and output data.
This submodule is intended to serve as a foundational part of flepiMoP's
infrastructure for conducting simulations and storing results.

Classes:
    TimeSetup: Handles simulation time frame.
    ModelInfo: Parses config file, holds model information, and manages file input/output.
"""

import datetime
import logging
import os
import pathlib
from typing import Literal

import confuse
import numba as nb
import numpy as np
import numpy.typing as npt
import pandas as pd

from . import (
    compartments,
    file_paths,
    parameters,
    seeding,
)
from .initial_conditions import initial_conditions_from_plugin
from .model_meta import ModelMeta
from .subpopulation_structure import SubpopulationStructure
from .time_setup import TimeSetup
from .utils import read_df, write_df


logger = logging.getLogger(__name__)


class ModelInfo:
    """
    Parses config file, holds model information, and manages file input/output.

    Attributes:
        nslots: Number of slots for MCMC.
        write_csv: Whether to write results to CSV files (default is False).
        write_parquet: Whether to write results to parquet files (default is False)
        first_sim_index: Index of first simulation (default is 1).
        seir_modifiers_scenario: seir_modifiers_scenario: SEIR modifier.
        outcome_modifiers_scenario: Outcomes modifier.
        setup_name: Name of setup (to override config, if applicable).
        time_setup: `TimeSetup` object (start/end dates of simulation, as pd.DatetimeIndex).
        ti: Initial time (time start).
        tf: Final time (fime finish).
        n_days: Number of days in simulation.
        dates: pd.DatetimeIndex sequence of dates that span simulation.
        subpop_struct: `SubpopulationStructure` object (info about subpops).
        nsubpops: Number of subpopulations in simulation.
        subpop_pop: NumPy array containing population of each subpop.
        mobility: Matrix with values representing movement of people between subpops.
        path_prefix: Prefix to paths where simulation data files are stored.
        seir_config: SEIR configuration info, if relevant for simulation.
        seir_modifiers_library: Modifiers for SEIR model, if relevant for simulation.
        parameters_config: Parameter information from config.
        initial_conditions_config: Initial conditions information from config.
        seeding_config: Seeding config, if relevant.
        parameters: `Parameter` object containing information about parameters.
        seeding: Seeding configuration information, if relevant.
        initial_conditions: Initial condition information for simulation.
        npi_config_seir: Non-pharmaceutical intervention configurations for SEIR, if relevant.
        compartments: `Compartments` object contianing information about compartments.
        outcomes_config: Outcomes configurations, if relevant.
        npi_config_outcomes: Non-pharmaceutical intervention outcome configurations, if relevant.
        outcome_modifiers_library: Outcome modifiers, pulled from config.
        in_run_id: ID for input run (generated if not specified).
        out_run_id: ID for outputr run (generated if not specified).
        in_prefix: Path prefix for input directory.
        out_prefix: Path prefix for output directory.
        inference_filename_prefix: Path prefix for inference files directory.
        inference_filepath_suffix: Path suffix for inference files directory.
        timestamp: Current datetime.
        extension: File extensions.
        config_filepath: Path to configuration file.

    Raises:
        ValueError:
            If provided configuration information is incompatible with expectations.
        ValueError:
            If non-existent sections are referenced.
        NotImplementedError:
            If an unimplemented feature is referenced.

    Config sections:
    ```
        subpop_setup                  # Always required
        compartments                  # Required if running seir
        parameters                    # required if running seir
        seir                          # Required if running seir
        initial_conditions            # One of seeding or initial_conditions is required when running seir
        seeding                       # One of seeding or initial_conditions is required when running seir
        outcomes                      # Required if running outcomes
        seir_modifiers                # Not required. If exists, every modifier will be applied to seir parameters
        outcomes_modifiers            # Not required. If exists, every modifier will be applied to outcomes
        inference                     # Required if running inference
    ```
    """

    def __init__(
        self,
        *,
        config,
        nslots=1,
        seir_modifiers_scenario=None,
        outcome_modifiers_scenario=None,
        path_prefix="",
        write_csv=False,
        write_parquet=False,
        first_sim_index=1,
        in_run_id=None,
        in_prefix=None,
        out_run_id=None,
        out_prefix=None,
        inference_filename_prefix="",
        inference_filepath_suffix="",
        setup_name=None,  # override config setup_name
        config_filepath="",
    ):
        """
        Initializes a `ModelInfo` object.

        Args:
            config: Config object.
            nslots: Number of slots for MCMC (default is 1).
            write_csv: Whether to write results to CSV files (default is False).
            write_parquet: Whether to write results to parquet files (default is False).
            first_sim_index : Index of first simulation (default is 1).
            seir_modifiers_scenario: SEIR modifier.
            outcome_modifiers_scenario: Outcomes modifier.
            setup_name: Name of setup (to override config, if applicable).
            path_prefix: Prefix to paths where simulation data files are stored.
            in_run_id: ID for input run (generated if not specified).
            out_run_id: ID for outputr run (generated if not specified).
            in_prefix: Path prefix for input directory.
            out_prefix: Path prefix for output directory.
            inference_filename_prefix: Path prefix for inference files directory.
            inference_filepath_suffix: Path suffix for inference files directory.
            config_filepath: Path to configuration file.
        """
        # Quick heuristic to check if the config is compatible
        # with this version of flepiMoP. Early exit if not.
        if config["interventions"].exists():
            msg = (
                "This config has an 'intervention' section, which is only "
                "compatible with a previous version (v1.1) of flepiMoP."
            )
            raise ValueError(msg)

        # Config filepath for plugins to reference
        self.config_filepath = config_filepath

        # Create a `ModelMeta` object to hold the core model metadata, then
        # assign to attributes of this instance to be backwards compatible.
        self.meta = ModelMeta.from_confuse_config(
            config=config,
            nslots=nslots,
            write_csv=write_csv,
            write_parquet=write_parquet,
            first_sim_index=first_sim_index,
            seir_modifiers_scenario=seir_modifiers_scenario,
            outcome_modifiers_scenario=outcome_modifiers_scenario,
            setup_name_=setup_name,
            path_prefix=path_prefix,
            in_run_id=in_run_id,
            in_prefix=in_prefix,
            out_run_id=out_run_id,
            out_prefix=out_prefix,
            inference_filename_prefix=inference_filename_prefix,
            inference_filepath_suffix=inference_filepath_suffix,
        )
        if any((write_csv, write_parquet)):
            self.meta.create_model_output_directories(
                (["seir", "spar", "snpi"] if config["seir"].exists() else [])
                + (["hosp", "hpar", "hnpi"] if config["outcomes"].exists() else []),
                "out",
            )
        for key, value in self.meta.model_dump().items():
            setattr(self, key, value)

        # 2. What about time:
        # Maybe group time_setup and subpop_struct into one argument for classes
        # make the import object first level attributes
        self.time_setup = TimeSetup(
            start_date=config["start_date"].as_date(), end_date=config["end_date"].as_date()
        )
        self.ti = self.time_setup.ti
        self.tf = self.time_setup.tf
        self.n_days = self.time_setup.n_days
        self.dates = self.time_setup.dates

        # 3. What about subpopulations
        self.path_prefix = pathlib.Path(path_prefix)
        self.subpop_struct = SubpopulationStructure.from_confuse_config(
            config["subpop_setup"], path_prefix=self.path_prefix
        )
        self.nsubpops = self.subpop_struct.nsubpops
        self.subpop_pop = self.subpop_struct.subpop_pop
        self.mobility = self.subpop_struct.mobility_matrix

        # 4. the SEIR structure
        self.seir_config = None
        self.seir_modifiers_library = None
        if config["seir"].exists():
            self.seir_config = config["seir"]
            if not self.seir_config["integration"].exists():
                self.seir_config["integration"]["method"].set("rk4")
                self.seir_config["integration"]["dt"].set(2)
            else:
                if not self.seir_config["integration"]["method"].exists():
                    self.seir_config["integration"]["method"].set("rk4")
                if not self.seir_config["integration"]["dt"].exists():
                    self.seir_config["integration"]["dt"].set(2)

            self.parameters_config = config["seir"]["parameters"]
            self.initial_conditions_config = (
                config["initial_conditions"]
                if config["initial_conditions"].exists()
                else None
            )
            self.seeding_config = config["seeding"] if config["seeding"].exists() else None

            if self.seeding_config is None and self.initial_conditions_config is None:
                logging.critical(
                    "The config has a seir: section but no initial_conditions: nor seeding: sections. At least one of them is needed"
                )
                # raise ValueError("The config has a seir: section but no initial_conditions: nor seeding: sections. At least one of them is needed")

            # Think if we really want to hold this up.
            self.parameters = parameters.Parameters(
                parameter_config=self.parameters_config,
                ti=self.ti,
                tf=self.tf,
                subpop_names=self.subpop_struct.subpop_names,
                path_prefix=self.path_prefix,
            )
            self.seeding = seeding.SeedingFactory(
                config=self.seeding_config, path_prefix=self.path_prefix
            )
            self.initial_conditions = initial_conditions_from_plugin(
                config["initial_conditions"],
                path_prefix=self.path_prefix,
                meta=self.meta,
                time_setup=self.time_setup,
            )

            # SEIR modifiers
            self.npi_config_seir = None
            if config["seir_modifiers"].exists():
                if config["seir_modifiers"]["scenarios"].exists():
                    self.npi_config_seir = config["seir_modifiers"]["modifiers"][
                        seir_modifiers_scenario
                    ]
                    self.seir_modifiers_library = config["seir_modifiers"][
                        "modifiers"
                    ].get()
                else:
                    self.seir_modifiers_library = config["seir_modifiers"][
                        "modifiers"
                    ].get()
                    raise NotImplementedError(
                        "This feature has not been implemented yet."
                    )  # TODO create a Stacked from all
            elif self.seir_modifiers_scenario is not None:
                raise ValueError(
                    "A `seir_modifiers_scenario` argument was provided to `ModelInfo` but there is no `seir_modifiers` section in the config."
                )
            else:
                logging.info("Running `ModelInfo` with seir but without SEIR Modifiers")

        elif self.seir_modifiers_scenario is not None:
            raise ValueError(
                "A `seir_modifiers_scenario` argument was provided to `ModelInfo` but there is no `seir` section in the config."
            )
        else:
            logging.critical("Running ModelInfo without SEIR")

        # really ugly references to the config globally here.
        self.compartments = (
            compartments.Compartments(
                seir_config=self.seir_config, compartments_config=config["compartments"]
            )
            if (config["compartments"].exists() and self.seir_config is not None)
            else None
        )

        # 5. Outcomes
        self.outcomes_config = config["outcomes"] if config["outcomes"].exists() else None
        self.npi_config_outcomes = None
        if self.outcomes_config is not None:
            if config["outcome_modifiers"].exists():
                if config["outcome_modifiers"]["scenarios"].exists():
                    self.npi_config_outcomes = config["outcome_modifiers"]["modifiers"][
                        self.outcome_modifiers_scenario
                    ]
                    self.outcome_modifiers_library = config["outcome_modifiers"][
                        "modifiers"
                    ].get()
                else:
                    self.outcome_modifiers_library = config["outcome_modifiers"][
                        "modifiers"
                    ].get()
                    raise NotImplementedError(
                        "This feature has not been implemented yet."
                    )  # TODO create a Stacked from all

            ## NEED TO IMPLEMENT THIS -- CURRENTLY CANNOT USE outcome modifiers
            elif self.outcome_modifiers_scenario is not None:
                if config["outcome_modifiers"].exists():
                    raise ValueError(
                        "A `outcome_modifiers_scenario` argument was provided to `ModelInfo` but there is no `outcome_modifiers` section in the config."
                    )
                else:
                    self.outcome_modifiers_scenario = None
            else:
                logging.info(
                    "Running `ModelInfo` with outcomes but without Outcomes Modifiers"
                )
        elif self.outcome_modifiers_scenario is not None:
            raise ValueError(
                "A `outcome_modifiers_scenario` argument was provided to `ModelInfo` but there is no `outcomes` section in the config."
            )
        else:
            logging.info("Running `ModelInfo` without outcomes.")

    def get_input_filename(self, ftype: str, sim_id: int, extension_override: str = ""):
        return self.path_prefix / self.get_filename(
            ftype=ftype,
            sim_id=sim_id,
            input=True,
            extension_override=extension_override,
        )

    def get_output_filename(self, ftype: str, sim_id: int, extension_override: str = ""):
        return self.path_prefix / self.get_filename(
            ftype=ftype,
            sim_id=sim_id,
            input=False,
            extension_override=extension_override,
        )

    def get_filename(
        self, ftype: str, sim_id: int, input: bool, extension_override: str = ""
    ):
        return self.path_prefix / file_paths.create_file_name(
            run_id=self.in_run_id if input else self.out_run_id,
            prefix=self.in_prefix if input else self.out_prefix,
            index=sim_id + self.first_sim_index - 1,
            ftype=ftype,
            extension=extension_override if extension_override else self.extension,
            inference_filepath_suffix=self.inference_filepath_suffix,
            inference_filename_prefix=self.inference_filename_prefix,
        )

    def get_setup_name(self):
        return self.setup_name

    def read_simID(
        self, ftype: str, sim_id: int, input: bool = True, extension_override: str = ""
    ):
        fname = self.get_filename(
            ftype=ftype,
            sim_id=sim_id,
            input=input,
            extension_override=extension_override,
        )
        # print(f"Readings {fname}")
        return read_df(fname=fname)

    def write_simID(
        self,
        ftype: str,
        sim_id: int,
        df: pd.DataFrame,
        input: bool = False,
        extension_override: str = "",
    ):
        fname = self.get_filename(
            ftype=ftype,
            sim_id=sim_id,
            input=input,
            extension_override=extension_override,
        )
        # create the directory if it does exists:
        os.makedirs(os.path.dirname(fname), exist_ok=True)

        # print(f"Writing {fname}")
        write_df(
            fname=fname,
            df=df,
        )
        return fname

    def get_seeding_data(self, sim_id: int) -> tuple[nb.typed.Dict, npt.NDArray[np.number]]:
        """
        Pull the seeding data for the info represented by this model info instance.

        Args:
            sim_id: The simulation ID to pull seeding data for.

        Returns:
            A tuple containing the seeding data dictionary and the seeding data array.

        See Also:
            `gempyor.seeding.Seeding.get_from_config`
        """
        return self.seeding.get_from_config(
            compartments=self.compartments,
            subpop_struct=self.subpop_struct,
            n_days=self.n_days,
            ti=self.ti,
            tf=self.tf,
            input_filename=(
                self.get_input_filename(
                    ftype=self.seeding_config["seeding_file_type"].get(),
                    sim_id=sim_id,
                    extension_override="csv",
                )
                if (
                    self.seeding_config is not None
                    and self.seeding_config["seeding_file_type"].exists()
                )
                else None
            ),
        )

    def get_engine(self) -> Literal["rk4", "euler", "stochastic"]:
        return self.seir_config["integration"]["method"].as_str()

    def get_initial_conditions_data(self, sim_id: int) -> npt.NDArray[np.int64]:
        """
        Pull the initial conditions data fro the info represented by this instance.

        Args:
            sim_id: The simulation ID to pull initial conditions data for.

        Returns:
            A two dimensional numpy array of integers with shape of
            (compartments, subpopulations).

        """
        return self.initial_conditions.get_initial_conditions(
            sim_id, self.compartments, self.subpop_struct
        )
