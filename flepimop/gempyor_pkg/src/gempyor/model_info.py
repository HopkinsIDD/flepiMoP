from distutils import extension
import pathlib
import re
import numpy as np
import pandas as pd
import datetime
import os
import scipy.sparse
import pyarrow as pa
import copy
from . import compartments
from . import parameters
from . import seeding_ic, subpopulation_structure
from .utils import config, read_df, write_df
from . import file_paths
import logging

logger = logging.getLogger(__name__)


class ModelInfo:
    """
    This class hold a full model setup.
    """

    def __init__(
        self,
        *,
        nslots,
        config,
        seir_modifiers_scenario=None,
        outcome_modifiers_scenario=None,
        write_csv=False,
        write_parquet=False,
        first_sim_index=1,
        in_run_id=None,
        in_prefix=None,
        out_run_id=None,
        out_prefix=None,
        stoch_traj_flag=False,
    ):
        # 1. Important global variables
        self.setup_name = config["name"].get() + "_" + str(seir_modifiers_scenario)
        self.nslots = nslots

        self.ti = config["start_date"].as_date()  ## we start at 00:00 on ti
        self.tf = config["end_date"].as_date()  ## we end on 23:59 on tf
        if self.tf <= self.ti:
            raise ValueError("tf (time to finish) is less than or equal to ti (time to start)")

        self.seir_modifiers_scenario = seir_modifiers_scenario
        self.npi_config_seir = config["seir_modifiers"]["settings"][seir_modifiers_scenario]
        self.seeding_config = config["seeding"]
        self.initial_conditions_config = config["initial_conditions"]
        self.parameters_config = config["seir"]["parameters"]
        self.outcomes_config = config["outcomes"] if config["outcomes"].exists() else None

        self.seir_config = config["seir"]
        self.write_csv = write_csv
        self.write_parquet = write_parquet
        self.first_sim_index = first_sim_index
        self.outcome_modifiers_scenario = outcome_modifiers_scenario

        spatial_config = config["subpop_setup"]
        spatial_base_path = config["data_path"].get()
        spatial_base_path = pathlib.Path(spatial_path_prefix + spatial_base_path)

        self.subpop_struct = subpopulation_structure.SubpopulationStructure(
                setup_name=config["setup_name"].get(),
                geodata_file=spatial_base_path / spatial_config["geodata"].get(),
                mobility_file=spatial_base_path / spatial_config["mobility"].get()
                if spatial_config["mobility"].exists()
                else None,
                subpop_pop_key="population",
                subpop_names_key="subpop",
            )
        self.n_days = (self.tf - self.ti).days + 1  # because we include s.ti and s.tf
        self.nsubpops = self.subpop_struct.nsubpops
        self.subpop_pop = self.subpop_struct.subpop_pop
        self.mobility = self.subpop_struct.mobility

        self.stoch_traj_flag = stoch_traj_flag

        # I'm not really sure if we should impose defaut or make setup really explicit and
        # have users pass
        if seir_config is None and config["seir"].exists():
            self.seir_config = config["seir"]

        # Set-up the integration method and the time step
        if config["seir"].exists() and (seir_config or parameters_config):
            if "integration" in self.seir_config.keys():
                if "method" in self.seir_config["integration"].keys():
                    self.integration_method = self.seir_config["integration"]["method"].get()
                    if self.integration_method == "best.current":
                        self.integration_method = "rk4.jit"
                    if self.integration_method == "rk4":
                        self.integration_method = "rk4.jit"
                    if self.integration_method not in ["rk4.jit", "legacy"]:
                        raise ValueError(f"Unknown integration method {self.integration_method}.")
                if "dt" in self.seir_config["integration"].keys() and self.dt is None:
                    self.dt = float(
                        eval(str(self.seir_config["integration"]["dt"].get()))
                    )  # ugly way to parse string and formulas
                elif self.dt is None:
                    self.dt = 2.0
            else:
                self.integration_method = "rk4.jit"
                if self.dt is None:
                    self.dt = 2.0
                logging.info(f"Integration method not provided, assuming type {self.integration_method}")
            if self.dt is not None:
                self.dt = float(self.dt)

            # Think if we really want to hold this up.
            self.parameters = parameters.Parameters(
                parameter_config=self.parameters_config,
                ti=self.ti,
                tf=self.tf,
                subpop_names=self.subpop_struct.subpop_names,
            )
            self.seedingAndIC = seeding_ic.SeedingAndIC(
                seeding_config=self.seeding_config,
                initial_conditions_config=self.initial_conditions_config,
            )
            # really ugly references to the config globally here.
            if config["compartments"].exists() and self.seir_config is not None:
                self.compartments = compartments.Compartments(
                    seir_config=self.seir_config, compartments_config=config["compartments"]
                )

        # 3. Outcomes
        self.npi_config_outcomes = None
        if self.outcomes_config:
            if config["outcomes_modifiers"].exists():
                if config["outcomes_modifiers"]["scenarios"].exists():
                    self.npi_config_outcomes = self.outcomes_config["outcomes_modifiers"]["modifiers"][self.outcome_modifiers_scenario]
                else:
                    self.npi_config_outcomes = config["outcomes_modifiers"]

        # 4. Inputs and outputs
        if in_run_id is None:
            in_run_id = file_paths.run_id()
        self.in_run_id = in_run_id

        if out_run_id is None:
            out_run_id = file_paths.run_id()
        self.out_run_id = out_run_id

        if in_prefix is None:
            in_prefix = f"model_output/{setup_name}/{in_run_id}/"
        self.in_prefix = in_prefix
        if out_prefix is None:
            out_prefix = f"model_output/{setup_name}/{seir_modifiers_scenario}/{out_run_id}/"
        self.out_prefix = out_prefix

        if self.write_csv or self.write_parquet:
            self.timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            ftypes = []
            if config["seir"].exists():
                ftypes.extend(["seir", "spar", "snpi"])
            if outcomes_config:
                ftypes.extend(["hosp", "hpar", "hnpi"])
            for ftype in ftypes:
                datadir = file_paths.create_dir_name(self.out_run_id, self.out_prefix, ftype)
                os.makedirs(datadir, exist_ok=True)

            if self.write_parquet and self.write_csv:
                print("Confused between reading .csv or parquet. Assuming input file is .parquet")
            if self.write_parquet:
                self.extension = "parquet"
            elif self.write_csv:
                self.extension = "csv"

    def get_input_filename(self, ftype: str, sim_id: int, extension_override: str = ""):
        return self.get_filename(
            ftype=ftype,
            sim_id=sim_id,
            input=True,
            extension_override=extension_override,
        )

    def get_output_filename(self, ftype: str, sim_id: int, extension_override: str = ""):
        return self.get_filename(
            ftype=ftype,
            sim_id=sim_id,
            input=False,
            extension_override=extension_override,
        )

    def get_filename(self, ftype: str, sim_id: int, input: bool, extension_override: str = ""):
        """return a CSP formated filename."""

        if extension_override:  # empty strings are Falsy
            extension = extension_override
        else:  # Constructed like this because in some test, extension is not defined
            extension = self.extension

        if input:
            run_id = self.in_run_id
            prefix = self.in_prefix
        else:
            run_id = self.out_run_id
            prefix = self.out_prefix

        fn = file_paths.create_file_name(
            run_id=run_id,
            prefix=prefix,
            index=sim_id + self.first_sim_index - 1,
            ftype=ftype,
            extension=extension,
        )
        return fn

    def read_simID(self, ftype: str, sim_id: int, input: bool = True, extension_override: str = ""):
        return read_df(
            fname=self.get_filename(
                ftype=ftype,
                sim_id=sim_id,
                input=input,
                extension_override=extension_override,
            )
        )

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
        write_df(
            fname=fname,
            df=df,
        )
        return fname
