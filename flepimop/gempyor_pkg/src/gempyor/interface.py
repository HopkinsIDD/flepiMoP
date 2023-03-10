#!/usr/bin/env python

##
# interface.py defines handlers to the gempyor epidemic module
# (both (SEIR) and the pipeline outcomes module (Outcomes))
# so they can be used from R for inference.
# R folks needs to define start a python, and set some variable as describe in the notebook
# This populate the namespace with four functions, with return value 0 if the
# function terminated successfully


import pathlib
from . import seir, setup, file_paths
from . import outcomes
from .utils import config, Timer, read_df, profile
import numpy as np
from concurrent.futures import ProcessPoolExecutor

### Logger configuration
import logging
import os
import functools
import multiprocessing as mp
import pandas as pd
import pyarrow.parquet as pq


logging.basicConfig(level=os.environ.get("FLEPI_LOGLEVEL", "INFO").upper())
logger = logging.getLogger()
handler = logging.StreamHandler()
# '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
formatter = logging.Formatter(
    " %(name)s :: %(levelname)-8s :: %(message)s"
    # "%(asctime)s [%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
)

handler.setFormatter(formatter)
# logger.addHandler(handler)


class InferenceSimulator:
    def __init__(
        self,
        config_path,
        run_id="test_run_id",
        prefix="test_prefix",
        first_sim_index=1,
        scenario="inference",
        deathrate="med",
        stoch_traj_flag=False,
        rng_seed=None,
        nslots=1,
        initialize=True,
        out_run_id=None,  # if out_run_id is different from in_run_id, fill this
        out_prefix=None,  # if out_prefix is different from in_prefix, fill this
        spatial_path_prefix="",  # in case the data folder is on another directory
    ):
        self.scenario = scenario
        self.deathrate = deathrate

        in_run_id = run_id
        if out_run_id is None:
            out_run_id = in_run_id
        in_prefix = prefix
        if out_prefix is None:
            out_prefix = in_prefix

        # Config prep
        config.clear()
        config.read(user=False)
        config.set_file(config_path)
        spatial_config = config["spatial_setup"]
        spatial_base_path = spatial_config["base_path"].get()
        spatial_base_path = pathlib.Path(spatial_path_prefix + spatial_base_path)

        np.random.seed(rng_seed)

        interactive = False
        write_csv = False
        write_parquet = True
        self.s = setup.Setup(
            setup_name=config["name"].get() + "_" + str(scenario),
            spatial_setup=setup.SpatialSetup(
                setup_name=spatial_config["setup_name"].get(),
                geodata_file=spatial_base_path / spatial_config["geodata"].get(),
                mobility_file=spatial_base_path / spatial_config["mobility"].get()
                if spatial_config["mobility"].exists()
                else None,
                popnodes_key=spatial_config["popnodes"].get(),
                nodenames_key=spatial_config["nodenames"].get(),
            ),
            nslots=nslots,
            npi_scenario=scenario,
            npi_config_seir=config["interventions"]["settings"][scenario],
            seeding_config=config["seeding"],
            initial_conditions_config=config["initial_conditions"],
            parameters_config=config["seir"]["parameters"],
            seir_config=config["seir"],
            outcomes_config=config["outcomes"] if config["outcomes"].exists() else None,
            outcomes_scenario=deathrate,
            ti=config["start_date"].as_date(),
            tf=config["end_date"].as_date(),
            interactive=interactive,
            write_csv=write_csv,
            write_parquet=write_parquet,
            dt=config["dt"].as_number(),
            first_sim_index=first_sim_index,
            in_run_id=in_run_id,
            in_prefix=in_prefix,
            out_run_id=out_run_id,
            out_prefix=out_prefix,
            stoch_traj_flag=stoch_traj_flag,
        )

        print(
            f"""  gempyor >> Running ***{'STOCHASTIC' if stoch_traj_flag else 'DETERMINISTIC'}*** simulation;\n"""
            f"""  gempyor >> Setup {self.s.setup_name}; index: {self.s.first_sim_index}; run_id: {in_run_id},\n"""
            f"""  gempyor >> prefix: {in_prefix};"""  # ti: {s.ti};??tf: {s.tf};
        )

        self.already_built = False  # whether we have already build the costly object we just build once.

    def update_prefix(self, new_prefix, new_out_prefix=None):
        self.s.in_prefix = new_prefix
        if new_out_prefix is None:
            self.s.out_prefix = new_prefix
        else:
            self.s.out_prefix = new_out_prefix

    def update_run_id(self, new_run_id, new_out_run_id=None):
        self.s.in_run_id = new_run_id
        if new_out_run_id is None:
            self.s.out_run_id = new_run_id
        else:
            self.s.out_run_id = new_out_run_id

    def one_simulation_legacy(self, sim_id2write: int, load_ID: bool = False, sim_id2load: int = None):
        sim_id2write = int(sim_id2write)
        if load_ID:
            sim_id2load = int(sim_id2load)
        with Timer(f">>> GEMPYOR onesim {'(loading file)' if load_ID else '(from config)'}"):
            with Timer("onerun_SEIR"):
                seir.onerun_SEIR(
                    sim_id2write=sim_id2write,
                    s=self.s,
                    load_ID=load_ID,
                    sim_id2load=sim_id2load,
                    config=config,
                )

            with Timer("onerun_OUTCOMES"):
                outcomes.onerun_delayframe_outcomes(
                    sim_id2write=sim_id2write,
                    s=self.s,
                    load_ID=load_ID,
                    sim_id2load=sim_id2load,
                )
        return 0

    # @profile()
    def one_simulation(
        self,
        sim_id2write: int,
        load_ID: bool = False,
        sim_id2load: int = None,
        parallel=False,
    ):
        sim_id2write = int(sim_id2write)
        if load_ID:
            sim_id2load = int(sim_id2load)

        with Timer(f">>> GEMPYOR onesim {'(loading file)' if load_ID else '(from config)'}"):
            if not self.already_built:
                self.outcomes_parameters = outcomes.read_parameters_from_config(self.s)

            npi_outcomes = None
            if parallel:
                with Timer("//things"):
                    with ProcessPoolExecutor(max_workers=max(mp.cpu_count(), 3)) as executor:
                        ret_seir = executor.submit(seir.build_npi_SEIR, self.s, load_ID, sim_id2load, config)
                        if self.s.npi_config_outcomes:
                            ret_outcomes = executor.submit(
                                outcomes.build_npi_Outcomes,
                                self.s,
                                load_ID,
                                sim_id2load,
                                config,
                            )
                        if not self.already_built:
                            ret_comparments = executor.submit(self.s.compartments.get_transition_array)

                # print("expections:", ret_seir.exception(), ret_outcomes.exception(), ret_comparments.exception())

                if not self.already_built:
                    (
                        self.unique_strings,
                        self.transition_array,
                        self.proportion_array,
                        self.proportion_info,
                    ) = ret_comparments.result()
                    self.already_built = True
                npi_seir = ret_seir.result()
                if self.s.npi_config_outcomes:
                    npi_outcomes = ret_outcomes.result()
            else:
                if not self.already_built:
                    (
                        self.unique_strings,
                        self.transition_array,
                        self.proportion_array,
                        self.proportion_info,
                    ) = self.s.compartments.get_transition_array()
                    self.already_built = True
                npi_seir = seir.build_npi_SEIR(s=self.s, load_ID=load_ID, sim_id2load=sim_id2load, config=config)
                if self.s.npi_config_outcomes:
                    npi_outcomes = outcomes.build_npi_Outcomes(
                        s=self.s,
                        load_ID=load_ID,
                        sim_id2load=sim_id2load,
                        config=config,
                    )
            self.debug_npi_seir = npi_seir
            self.debug_npi_outcomes = npi_outcomes
            ### Run every time:
            with Timer("SEIR.parameters"):
                # Draw or load parameters
                p_draw = self.get_seir_parameters(load_ID=load_ID, sim_id2load=sim_id2load)

                # reduce them
                parameters = self.s.parameters.parameters_reduce(p_draw, npi_seir)

                # Parse them
                parsed_parameters = self.s.compartments.parse_parameters(
                    parameters, self.s.parameters.pnames, self.unique_strings
                )
                self.debug_p_draw = p_draw
                self.debug_parameters = parameters
                self.debug_parsed_parameters = parsed_parameters

            with Timer("onerun_SEIR.seeding"):
                if load_ID:
                    initial_conditions = self.s.seedingAndIC.load_ic(sim_id2load, setup=self.s)
                    seeding_data, seeding_amounts = self.s.seedingAndIC.load_seeding(sim_id2load, setup=self.s)
                else:
                    initial_conditions = self.s.seedingAndIC.draw_ic(sim_id2write, setup=self.s)
                    seeding_data, seeding_amounts = self.s.seedingAndIC.draw_seeding(sim_id2write, setup=self.s)
                self.debug_seeding_date = seeding_data
                self.debug_seeding_amounts = seeding_amounts

            with Timer("SEIR.compute"):
                states = seir.steps_SEIR(
                    self.s,
                    parsed_parameters,
                    self.transition_array,
                    self.proportion_array,
                    self.proportion_info,
                    initial_conditions,
                    seeding_data,
                    seeding_amounts,
                )
                self.debug_states = states

            with Timer("SEIR.postprocess"):
                if self.s.write_csv or self.s.write_parquet:
                    out_df = seir.postprocess_and_write(sim_id2write, self.s, states, p_draw, npi_seir, seeding_data)
                    self.debug_out_df = out_df

            loaded_values = None
            if load_ID:
                loaded_values = self.s.read_simID(ftype="hpar", sim_id=sim_id2load)
                self.debug_loaded_values = loaded_values

            # Compute outcomes
            with Timer("onerun_delayframe_outcomes.compute"):
                outcomes_df, hpar_df = outcomes.compute_all_multioutcomes(
                    s=self.s,
                    sim_id2write=sim_id2write,
                    parameters=self.outcomes_parameters,
                    loaded_values=loaded_values,
                    npi=npi_outcomes,
                )
                self.debug_outcomes_df = outcomes_df
                self.debug_hpar_df = hpar_df

            with Timer("onerun_delayframe_outcomes.postprocess"):
                outcomes.postprocess_and_write(
                    sim_id=sim_id2write,
                    s=self.s,
                    outcomes=outcomes_df,
                    hpar=hpar_df,
                    npi=npi_outcomes,
                )
        return 0

    def plot_transition_graph(self, output_file="transition_graph", source_filters=[], destination_filters=[]):
        self.s.compartments.plot(
            output_file=output_file,
            source_filters=source_filters,
            destination_filters=destination_filters,
        )

    def get_outcome_npi(self, load_ID=False, sim_id2load=None, bypass_DF=None, bypass_FN=None):
        npi_outcomes = None
        if self.s.npi_config_outcomes:
            npi_outcomes = outcomes.build_npi_Outcomes(
                s=self.s,
                load_ID=load_ID,
                sim_id2load=sim_id2load,
                config=config,
                bypass_DF=bypass_DF,
                bypass_FN=bypass_FN,
            )
        return npi_outcomes

    def get_seir_npi(self, load_ID=False, sim_id2load=None, bypass_DF=None, bypass_FN=None):
        npi_seir = seir.build_npi_SEIR(
            s=self.s,
            load_ID=load_ID,
            sim_id2load=sim_id2load,
            config=config,
            bypass_DF=bypass_DF,
            bypass_FN=bypass_FN,
        )
        return npi_seir

    def get_seir_parameters(self, load_ID=False, sim_id2load=None, bypass_DF=None, bypass_FN=None):
        param_df = None
        if bypass_DF is not None:
            param_df = bypass_DF
        elif bypass_FN is not None:
            param_df = read_df(fname=bypass_FN)
        elif load_ID == True:
            param_df = self.s.read_simID(ftype="spar", sim_id=sim_id2load)

        if param_df is not None:
            p_draw = self.s.parameters.parameters_load(
                param_df=param_df,
                n_days=self.s.n_days,
                nnodes=self.s.nnodes,
            )
        else:
            p_draw = self.s.parameters.parameters_quick_draw(n_days=self.s.n_days, nnodes=self.s.nnodes)
        return p_draw

    def get_seir_parametersDF(self, load_ID=False, sim_id2load=None, bypass_DF=None, bypass_FN=None):
        p_draw = self.get_seir_parameters(
            load_ID=load_ID,
            sim_id2load=sim_id2load,
            bypass_DF=bypass_DF,
            bypass_FN=bypass_FN,
        )
        return self.s.parameters.getParameterDF(p_draw=p_draw)

    def get_seir_parameter_reduced(
        self,
        npi_seir,
        p_draw=None,
        load_ID=False,
        sim_id2load=None,
        bypass_DF=None,
        bypass_FN=None,
    ):
        if p_draw is None:
            p_draw = self.get_seir_parameters(
                load_ID=load_ID,
                sim_id2load=sim_id2load,
                bypass_DF=bypass_DF,
                bypass_FN=bypass_FN,
            )

        parameters = self.s.parameters.parameters_reduce(p_draw, npi_seir)

        full_df = pd.DataFrame()
        for i, geoid in enumerate(self.s.spatset.nodenames):
            a = pd.DataFrame(
                parameters[:, :, i].T,
                columns=self.s.parameters.pnames,
                index=pd.date_range(self.s.ti, self.s.tf, freq="D"),
            )
            a["geoid"] = geoid
            full_df = pd.concat([full_df, a])

        # for R, duplicate names are not allowed in index:
        full_df["date"] = full_df.index
        full_df = full_df.reset_index(drop=True)

        return full_df


def paramred_parallel(run_spec, snpi_fn):
    config_filepath = run_spec["config"]
    gempyor_simulator = InferenceSimulator(
        config_path=config_filepath,
        run_id="test_run_id",
        prefix="test_prefix/",
        first_sim_index=1,
        scenario="inference",  # NPIs scenario to use
        deathrate="med",  # Outcome scenario to use
        stoch_traj_flag=False,
        spatial_path_prefix=run_spec["geodata"],  # prefix where to find the folder indicated in spatial_setup$
    )

    snpi = pq.read_table(snpi_fn).to_pandas()

    npi_seir = gempyor_simulator.get_seir_npi(bypass_DF=snpi)

    # params_draw_df = gempyor_simulator.get_seir_parametersDF()  # could also accept (load_ID=True, sim_id2load=XXX) or (bypass_DF=<some_spar_df>) or (bypass_FN=<some_spar_filename>)
    params_draw_arr = gempyor_simulator.get_seir_parameters(
        bypass_FN=snpi_fn.replace("snpi", "spar")
    )  # could also accept (load_ID=True, sim_id2load=XXX) or (bypass_DF=<some_spar_df>) or (bypass_FN=<some_spar_filename>)
    param_reduc_from = gempyor_simulator.get_seir_parameter_reduced(npi_seir=npi_seir, p_draw=params_draw_arr)

    return param_reduc_from


def paramred_parallel_config(run_spec, dummy):
    config_filepath = run_spec["config"]
    gempyor_simulator = InferenceSimulator(
        config_path=config_filepath,
        run_id="test_run_id",
        prefix="test_prefix/",
        first_sim_index=1,
        scenario="inference",  # NPIs scenario to use
        deathrate="med",  # Outcome scenario to use
        stoch_traj_flag=False,
        spatial_path_prefix=run_spec["geodata"],  # prefix where to find the folder indicated in spatial_setup$
    )

    npi_seir = gempyor_simulator.get_seir_npi()

    params_draw_arr = (
        gempyor_simulator.get_seir_parameters()
    )  # could also accept (load_ID=True, sim_id2load=XXX) or (bypass_DF=<some_spar_df>) or (bypass_FN=<some_spar_filename>)
    param_reduc_from = gempyor_simulator.get_seir_parameter_reduced(npi_seir=npi_seir, p_draw=params_draw_arr)

    return param_reduc_from
