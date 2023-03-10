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
from . import seeding_ic
from .utils import config, read_df, write_df
from . import file_paths
import logging

logger = logging.getLogger(__name__)


class Setup:
    """
    This class hold a setup model setup.
    """

    def __init__(
        self,
        *,
        setup_name,
        spatial_setup,
        nslots,
        ti,  # time to start
        tf,  # time to finish
        npi_scenario=None,
        config_version=None,
        npi_config_seir={},
        seeding_config={},
        initial_conditions_config={},
        parameters_config={},
        seir_config={},
        outcomes_config={},
        outcomes_scenario=None,
        interactive=True,
        write_csv=False,
        write_parquet=False,
        dt=1 / 6,  # step size, in days
        first_sim_index=1,
        in_run_id=None,
        in_prefix=None,
        out_run_id=None,
        out_prefix=None,
        stoch_traj_flag=False,
    ):

        # 1. Important global variables
        self.setup_name = setup_name
        self.nslots = nslots
        self.dt = float(dt)
        self.ti = ti  ## we start at 00:00 on ti
        self.tf = tf  ## we end on 23:59 on tf
        if self.tf <= self.ti:
            raise ValueError("tf (time to finish) is less than or equal to ti (time to start)")
        self.npi_scenario = npi_scenario
        self.npi_config_seir = npi_config_seir
        self.seeding_config = seeding_config
        self.initial_conditions_config = initial_conditions_config
        self.parameters_config = parameters_config
        self.outcomes_config = outcomes_config

        self.seir_config = seir_config
        self.interactive = interactive
        self.write_csv = write_csv
        self.write_parquet = write_parquet
        self.first_sim_index = first_sim_index
        self.outcomes_scenario = outcomes_scenario

        self.spatset = spatial_setup
        self.n_days = (self.tf - self.ti).days + 1  # because we include s.ti and s.tf
        self.nnodes = self.spatset.nnodes
        self.popnodes = self.spatset.popnodes
        self.mobility = self.spatset.mobility

        self.stoch_traj_flag = stoch_traj_flag

        # SEIR part
        if config["seir"].exists() and (seir_config or parameters_config):
            if "integration_method" in self.seir_config.keys():
                self.integration_method = self.seir_config["integration_method"].get()
                if self.integration_method == "best.current":
                    self.integration_method = "rk4.jit"
                if self.integration_method == "rk4":
                    self.integration_method = "rk4.jit"
                if self.integration_method not in ["rk4.jit", "legacy"]:
                    raise ValueError(f"Unknow integration method {self.integration_method}.")
            else:
                self.integration_method = "rk4.jit"
                logging.info(f"Integration method not provided, assuming type {self.integration_method}")

            if config_version is None:
                if "compartments" in self.seir_config.keys():
                    config_version = "v2"
                else:
                    config_version = "old"

                logging.debug(f"Config version not provided, infering type {config_version}")

            if config_version != "old" and config_version != "v2":
                raise ValueError(
                    f"Configuration version unknown: {config_version}. "
                    f"Should be either non-specified (default: 'old'), or set to 'old' or 'v2'."
                )

            # Think if we really want to hold this up.
            self.parameters = parameters.Parameters(
                parameter_config=self.parameters_config,
                config_version=config_version,
                ti=self.ti,
                tf=self.tf,
                nodenames=self.spatset.nodenames,
            )
            self.seedingAndIC = seeding_ic.SeedingAndIC(
                seeding_config=self.seeding_config,
                initial_conditions_config=self.initial_conditions_config,
            )
            self.compartments = compartments.Compartments(self.seir_config)

        # 3. Outcomes
        self.npi_config_outcomes = None
        if self.outcomes_config:
            if self.outcomes_config["interventions"]["settings"][self.outcomes_scenario].exists():
                self.npi_config_outcomes = self.outcomes_config["interventions"]["settings"][self.outcomes_scenario]

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
            out_prefix = f"model_output/{setup_name}/{npi_scenario}/{out_run_id}/"
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


class SpatialSetup:
    def __init__(self, *, setup_name, geodata_file, mobility_file, popnodes_key, nodenames_key):
        self.setup_name = setup_name
        self.data = pd.read_csv(geodata_file, converters={nodenames_key: lambda x: str(x)})  # geoids and populations
        self.nnodes = len(self.data)  # K = # of locations

        # popnodes_key is the name of the column in geodata_file with populations
        if popnodes_key not in self.data:
            raise ValueError(f"popnodes_key: {popnodes_key} does not correspond to a column in geodata.")
        self.popnodes = self.data[popnodes_key].to_numpy()  # population
        if len(np.argwhere(self.popnodes == 0)):
            raise ValueError(
                f"There are {len(np.argwhere(self.popnodes == 0))} nodes with population zero, this is not supported."
            )

        # nodenames_key is the name of the column in geodata_file with geoids
        if nodenames_key not in self.data:
            raise ValueError(f"nodenames_key: {nodenames_key} does not correspond to a column in geodata.")
        self.nodenames = self.data[nodenames_key].tolist()
        if len(self.nodenames) != len(set(self.nodenames)):
            raise ValueError(f"There are duplicate nodenames in geodata.")

        if mobility_file is not None:
            mobility_file = pathlib.Path(mobility_file)
            if mobility_file.suffix == ".txt":
                print("Mobility files as matrices are not recommended. Please switch soon to long form csv files.")
                self.mobility = scipy.sparse.csr_matrix(
                    np.loadtxt(mobility_file), dtype=int
                )  # K x K matrix of people moving
                # Validate mobility data
                if self.mobility.shape != (self.nnodes, self.nnodes):
                    raise ValueError(
                        f"mobility data must have dimensions of length of geodata ({self.nnodes}, {self.nnodes}). Actual: {self.mobility.shape}"
                    )

            elif mobility_file.suffix == ".csv":
                mobility_data = pd.read_csv(mobility_file, converters={"ori": str, "dest": str})
                nn_dict = {v: k for k, v in enumerate(self.nodenames)}
                mobility_data["ori_idx"] = mobility_data["ori"].apply(nn_dict.__getitem__)
                mobility_data["dest_idx"] = mobility_data["dest"].apply(nn_dict.__getitem__)
                if any(mobility_data["ori_idx"] == mobility_data["dest_idx"]):
                    raise ValueError(
                        f"Mobility fluxes with same origin and destination in long form matrix. This is not supported"
                    )

                self.mobility = scipy.sparse.coo_matrix(
                    (mobility_data.amount, (mobility_data.ori_idx, mobility_data.dest_idx)),
                    shape=(self.nnodes, self.nnodes),
                    dtype=int,
                ).tocsr()

            elif mobility_file.suffix == ".npz":
                self.mobility = scipy.sparse.load_npz(mobility_file).astype(int)
                # Validate mobility data
                if self.mobility.shape != (self.nnodes, self.nnodes):
                    raise ValueError(
                        f"mobility data must have dimensions of length of geodata ({self.nnodes}, {self.nnodes}). Actual: {self.mobility.shape}"
                    )
            else:
                raise ValueError(
                    f"Mobility data must either be a .csv file in longform (recommended) or a .txt matrix file. Got {mobility_file}"
                )

            # Make sure mobility values <= the population of src node
            tmp = (self.mobility.T - self.popnodes).T
            tmp[tmp < 0] = 0
            if tmp.any():
                rows, cols, values = scipy.sparse.find(tmp)
                errmsg = ""
                for r, c, v in zip(rows, cols, values):
                    errmsg += f"\n({r}, {c}) = {self.mobility[r, c]} > population of '{self.nodenames[r]}' = {self.popnodes[r]}"
                raise ValueError(
                    f"The following entries in the mobility data exceed the source node populations in geodata:{errmsg}"
                )

            tmp = self.popnodes - np.squeeze(np.asarray(self.mobility.sum(axis=1)))
            tmp[tmp > 0] = 0
            if tmp.any():
                (row,) = np.where(tmp)
                errmsg = ""
                for r in row:
                    errmsg += f"\n sum accross row {r} exceed population of node '{self.nodenames[r]}' ({self.popnodes[r]}), by {-tmp[r]}"
                raise ValueError(
                    f"The following rows in the mobility data exceed the source node populations in geodata:{errmsg}"
                )
        else:
            logging.critical("No mobility matrix specified -- assuming no one moves")
            self.mobility = scipy.sparse.csr_matrix(np.zeros((self.nnodes, self.nnodes)), dtype=int)
