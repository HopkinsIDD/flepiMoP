import pandas as pd
import datetime, os, logging, pathlib
from . import seeding, subpopulation_structure, parameters, compartments, file_paths, initial_conditions
from .utils import read_df, write_df

logger = logging.getLogger(__name__)


class ModelInfo:
    # TODO: update this documentation add explaination about the construction of ModelInfo
    """
    Parse config and hold some results, with main config sections.
    ```
        # subpop_setup       # Always required
        # compartments        # Required if running seir
        # seir                # Required if running seir
        # initial_conditions  # One of seeding or initial_conditions is required when running seir
        # seeding             # One of seeding or initial_conditions is required when running seir
        # outcomes            # Required if running outcomes
        # seir_modifiers      # Not required. If exists, every modifier will be applied to seir parameters
        # outcome_modifiers  # Not required. If exists, every modifier will be applied to outcomes parameters
        # inference           # Required if running inference
    ```
    """

    def __init__(
        self,
        *,
        config,
        nslots=1,
        seir_modifiers_scenario=None,
        outcome_modifiers_scenario=None,
        spatial_path_prefix="",
        write_csv=False,
        write_parquet=False,
        first_sim_index=1,
        in_run_id=None,
        in_prefix=None,
        out_run_id=None,
        out_prefix=None,
        stoch_traj_flag=False,
        inference_filename_prefix="",
        inference_filepath_suffix="",
        setup_name=None,  # override config setup_name
    ):
        self.nslots = nslots
        self.write_csv = write_csv
        self.write_parquet = write_parquet
        self.first_sim_index = first_sim_index
        self.stoch_traj_flag = stoch_traj_flag

        self.seir_modifiers_scenario = seir_modifiers_scenario
        self.outcome_modifiers_scenario = outcome_modifiers_scenario

        # Auto-detect old config
        if config["interventions"].exists():
            raise ValueError("""This config has an intervention section, and has been written for a previous version of flepiMoP/COVIDScenarioPipeline \
                             Please use flepiMoP Version 1.1 (Commit SHA: 0c30c23937dd496d33c2b9fa7c6edb198ad80dac) to run this config. \
                             (use git checkout v1.1 inside the flepiMoP directory)""")

        # 1. Create a setup name that contains every scenario.
        if setup_name is None:
            self.setup_name = config["name"].get()
            if self.seir_modifiers_scenario is not None:
                self.setup_name += "_" + str(self.seir_modifiers_scenario)
            if self.outcome_modifiers_scenario is not None:
                self.setup_name += "_" + str(self.outcome_modifiers_scenario)
        else:
            self.setup_name = setup_name

        # 2. What about time:
        self.ti = config["start_date"].as_date()  ## we start at 00:00 on ti
        self.tf = config["end_date"].as_date()  ## we end on 23:59 on tf
        if self.tf <= self.ti:
            raise ValueError("tf (time to finish) is less than or equal to ti (time to start)")
        self.n_days = (self.tf - self.ti).days + 1  # because we include ti and tf

        # 3. What about subpopulations
        spatial_config = config["subpop_setup"]
        if config["data_path"].exists():
            raise ValueError("The config has a data_path section. This is no longer supported.")
        spatial_base_path = pathlib.Path(spatial_path_prefix)

        self.subpop_struct = subpopulation_structure.SubpopulationStructure(
            setup_name=config["setup_name"].get(),
            geodata_file=spatial_base_path / spatial_config["geodata"].get(),
            mobility_file=spatial_base_path / spatial_config["mobility"].get()
            if spatial_config["mobility"].exists()
            else None,
            subpop_pop_key="population",
            subpop_names_key="subpop",
        )
        self.nsubpops = self.subpop_struct.nsubpops
        self.subpop_pop = self.subpop_struct.subpop_pop
        self.mobility = self.subpop_struct.mobility

        # 4. the SEIR structure
        if config["seir"].exists():
            self.seir_config = config["seir"]
            self.parameters_config = config["seir"]["parameters"]
            self.initial_conditions_config = (
                config["initial_conditions"] if config["initial_conditions"].exists() else None
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
            )
            self.seeding = seeding.SeedingFactory(config = self.seeding_config)
            self.initial_conditions = initial_conditions.InitialConditionsFactory(config = self.initial_conditions_config)
            # really ugly references to the config globally here.
            if config["compartments"].exists() and self.seir_config is not None:
                self.compartments = compartments.Compartments(
                    seir_config=self.seir_config, compartments_config=config["compartments"]
                )

            # SEIR modifiers
            self.npi_config_seir = None
            if config["seir_modifiers"].exists():
                if config["seir_modifiers"]["scenarios"].exists():
                    self.npi_config_seir = config["seir_modifiers"]["modifiers"][seir_modifiers_scenario]
                    self.seir_modifiers_library = config["seir_modifiers"]["modifiers"].get()
                else:
                    self.seir_modifiers_library = config["seir_modifiers"]["modifiers"].get()
                    raise ValueError("Not implemented yet")  # TODO create a Stacked from all
            elif self.seir_modifiers_scenario is not None:
                raise ValueError(
                    "An seir modifiers scenario was provided to ModelInfo but no 'seir_modifiers' sections in config"
                )
            else:
                logging.info("Running ModelInfo with seir but without SEIR Modifiers")

        elif self.seir_modifiers_scenario is not None:
            raise ValueError("A seir modifiers scenario was provided to ModelInfo but no 'seir:' sections in config")
        else:
            logging.critical("Running ModelInfo without SEIR")

        # 5. Outcomes
        if config["outcomes"].exists():
            self.outcomes_config = config["outcomes"] if config["outcomes"].exists() else None
            self.npi_config_outcomes = None
            if config["outcome_modifiers"].exists():
                if config["outcome_modifiers"]["scenarios"].exists():
                    self.npi_config_outcomes = config["outcome_modifiers"]["modifiers"][self.outcome_modifiers_scenario]
                    self.outcome_modifiers_library = config["outcome_modifiers"]["modifiers"].get()
                else:
                    self.outcome_modifiers_library = config["outcome_modifiers"]["modifiers"].get()
                    raise ValueError("Not implemented yet")  # TODO create a Stacked from all

            ## NEED TO IMPLEMENT THIS -- CURRENTLY CANNOT USE outcome modifiers
            elif self.outcome_modifiers_scenario is not None:
                if config["outcome_modifiers"].exists():
                    raise ValueError(
                        "An outcome modifiers scenario was provided to ModelInfo but no 'outcome_modifiers' sections in config"
                    )
                else:
                    self.outcome_modifiers_scenario = None
            else:
                logging.info("Running ModelInfo with outcomes but without Outcomes Modifiers")
        elif self.outcome_modifiers_scenario is not None:
            raise ValueError(
                "An outcome modifiers scenario was provided to ModelInfo but no 'outcomes:' sections in config"
            )
        else:
            logging.info("Running ModelInfo without Outcomes")

        # 6. Inputs and outputs
        if in_run_id is None:
            in_run_id = file_paths.run_id()
        self.in_run_id = in_run_id

        if out_run_id is None:
            out_run_id = in_run_id
        self.out_run_id = out_run_id

        if in_prefix is None:
            in_prefix = f"{self.setup_name}/{self.in_run_id}/"
        self.in_prefix = in_prefix
        if out_prefix is None:
            out_prefix = f"{self.setup_name}/{self.out_run_id}/"
        self.out_prefix = out_prefix

        # make the inference paths:
        self.inference_filename_prefix = inference_filename_prefix
        self.inference_filepath_suffix = inference_filepath_suffix

        if self.write_csv or self.write_parquet:
            self.timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            ftypes = []
            if config["seir"].exists():
                ftypes.extend(["seir", "spar", "snpi"])
            if config["outcomes"].exists():
                ftypes.extend(["hosp", "hpar", "hnpi"])
            for ftype in ftypes:
                datadir = file_paths.create_dir_name(
                    run_id=self.out_run_id,
                    prefix=self.out_prefix,
                    ftype=ftype,
                    inference_filename_prefix=inference_filename_prefix,
                    inference_filepath_suffix=inference_filepath_suffix,
                )
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
            inference_filepath_suffix=self.inference_filepath_suffix,
            inference_filename_prefix=self.inference_filename_prefix,
            ftype=ftype,
            extension=extension,
        )
        return fn

    def get_setup_name(self):
        return self.setup_name

    def read_simID(self, ftype: str, sim_id: int, input: bool = True, extension_override: str = ""):
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
        # print(f"Writing {fname}")
        write_df(
            fname=fname,
            df=df,
        )
        return fname
