#!/usr/bin/env python

##
# inference.py defines handlers to the gempyor epidemic module
# (both (SEIR) and the pipeline outcomes module (Outcomes))
# so they can be used from R for inference.
# R folks needs to define start a python, and set some variable as describe in the notebook
# This populate the namespace with four functions, with return value 0 if the
# function terminated successfully


from . import seir, model_info
from . import outcomes, file_paths
from .utils import config, Timer, read_df, as_list
import numpy as np
from concurrent.futures import ProcessPoolExecutor

# Logger configuration
import logging
import os
import multiprocessing as mp
import pandas as pd
import pyarrow.parquet as pq
import xarray as xr
import numba as nb


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

from . import seir, model_info
from . import outcomes
from .utils import config, Timer, read_df
import numpy as np
from concurrent.futures import ProcessPoolExecutor

# Logger configuration
import logging
import os
import multiprocessing as mp
import pandas as pd
import pyarrow.parquet as pq
import xarray as xr
import numba as nb
import copy
import matplotlib.pyplot as plt
import seaborn as sns
import confuse
from . import inference_parameter, logloss, statistics

# TODO: should be able to draw e.g from an initial condition folder buuut keep the draw as a blob
# so it is saved by emcee, so I can build a posterio


# TODO: there is way to many of these functions, merge with the R inference.py implementation to avoid code duplication
def simulation_atomic(
    *,
    snpi_df_in,
    hnpi_df_in,
    modinf,
    p_draw,
    unique_strings,
    transition_array,
    proportion_array,
    proportion_info,
    initial_conditions,
    seeding_data,
    seeding_amounts,
    outcomes_parameters,
    save=False,
):
    # We need to reseed because subprocess inherit of the same random generator state.
    np.random.seed(int.from_bytes(os.urandom(4), byteorder="little"))
    random_id = np.random.randint(0, 1e8)

    npi_seir = seir.build_npi_SEIR(modinf=modinf, load_ID=False, sim_id2load=None, config=config, bypass_DF=snpi_df_in)

    if modinf.npi_config_outcomes:
        npi_outcomes = outcomes.build_outcome_modifiers(
            modinf=modinf, load_ID=False, sim_id2load=None, config=config, bypass_DF=hnpi_df_in
        )
    else:
        npi_outcomes = None

    # reduce them
    parameters = modinf.parameters.parameters_reduce(p_draw, npi_seir)
    # Parse them
    parsed_parameters = modinf.compartments.parse_parameters(parameters, modinf.parameters.pnames, unique_strings)

    # Convert the seeding data dictionnary to a numba dictionnary
    seeding_data_nbdict = nb.typed.Dict.empty(key_type=nb.types.unicode_type, value_type=nb.types.int64[:])

    for k, v in seeding_data.items():
        seeding_data_nbdict[k] = np.array(v, dtype=np.int64)

    # Compute the SEIR simulation
    states = seir.steps_SEIR(
        modinf,
        parsed_parameters,
        transition_array,
        proportion_array,
        proportion_info,
        initial_conditions,
        seeding_data_nbdict,
        seeding_amounts,
    )

    # Compute outcomes
    outcomes_df, hpar_df = outcomes.compute_all_multioutcomes(
        modinf=modinf,
        sim_id2write=0,
        parameters=outcomes_parameters,
        loaded_values=None,
        npi=npi_outcomes,
        bypass_seir_xr=states,
    )
    outcomes_df, hpar, hnpi = outcomes.postprocess_and_write(
        sim_id=random_id,
        modinf=modinf,
        outcomes_df=outcomes_df,
        hpar=hpar_df,
        npi=npi_outcomes,
        write=save,
    )
    # needs to be after write... because parquet write discard the index.
    outcomes_df = outcomes_df.set_index("date")  # after writing

    return outcomes_df


def get_static_arguments(modinf):
    """
    Get the static arguments for the log likelihood function, these are the same for all walkers
    """

    real_simulation = False
    (
        unique_strings,
        transition_array,
        proportion_array,
        proportion_info,
    ) = modinf.compartments.get_transition_array()

    outcomes_parameters = outcomes.read_parameters_from_config(modinf)
    npi_seir = seir.build_npi_SEIR(modinf=modinf, load_ID=False, sim_id2load=None, config=config)
    if modinf.npi_config_outcomes:
        npi_outcomes = outcomes.build_outcome_modifiers(
            modinf=modinf,
            load_ID=False,
            sim_id2load=None,
            config=config,
        )
    else:
        npi_outcomes = None

    p_draw = modinf.parameters.parameters_quick_draw(n_days=modinf.n_days, nsubpops=modinf.nsubpops)

    initial_conditions = modinf.initial_conditions.get_from_config(sim_id=0, modinf=modinf)
    seeding_data, seeding_amounts = modinf.seeding.get_from_config(sim_id=0, modinf=modinf)

    # reduce them
    parameters = modinf.parameters.parameters_reduce(p_draw, npi_seir)
    # Parse them
    parsed_parameters = modinf.compartments.parse_parameters(parameters, modinf.parameters.pnames, unique_strings)

    if real_simulation:
        states = seir.steps_SEIR(
            modinf,
            parsed_parameters,
            transition_array,
            proportion_array,
            proportion_info,
            initial_conditions,
            seeding_data,
            seeding_amounts,
        )
    else:
        # To avoid doing a simulation, but since we still need to run outcomes to get the hpar file,
        # we create a fake state array is all zero
        compartment_coords = {}
        compartment_df = modinf.compartments.get_compartments_explicitDF()
        # Iterate over columns of the DataFrame and populate the dictionary
        for column in compartment_df.columns:
            compartment_coords[column] = ("compartment", compartment_df[column].tolist())

        coords = dict(
            date=pd.date_range(modinf.ti, modinf.tf, freq="D"),
            **compartment_coords,
            subpop=modinf.subpop_struct.subpop_names,
        )

        zeros = np.zeros((len(coords["date"]), len(coords["mc_name"][1]), len(coords["subpop"])))
        states = xr.Dataset(
            data_vars=dict(
                prevalence=(["date", "compartment", "subpop"], zeros),
                incidence=(["date", "compartment", "subpop"], zeros),
            ),
            coords=coords,
            attrs=dict(
                description="Dynamical simulation results", run_id=modinf.in_run_id
            ),  # TODO add more information
        )

    snpi_df_ref = npi_seir.getReductionDF()

    outcomes_df, hpar_df = outcomes.compute_all_multioutcomes(
        modinf=modinf,
        sim_id2write=0,
        parameters=outcomes_parameters,
        loaded_values=None,
        npi=npi_outcomes,
        bypass_seir_xr=states,
    )
    outcomes_df_ref, hpar_ref, hnpi_df_ref = outcomes.postprocess_and_write(
        sim_id=0,
        modinf=modinf,
        outcomes_df=outcomes_df,
        hpar=hpar_df,
        npi=npi_outcomes,
        write=False,
    )
    outcomes_df_ref = outcomes_df_ref.set_index("date")

    # need to convert to numba dict to python dict so it is pickable
    seeding_data = dict(seeding_data)
    static_sim_arguments = {
        "snpi_df_ref": snpi_df_ref,
        "hnpi_df_ref": hnpi_df_ref,
        "p_draw": p_draw,
        "unique_strings": unique_strings,
        "transition_array": transition_array,
        "proportion_array": proportion_array,
        "proportion_info": proportion_info,
        "initial_conditions": initial_conditions,
        "seeding_data": seeding_data,
        "seeding_amounts": seeding_amounts,
        "outcomes_parameters": outcomes_parameters,
    }

    return static_sim_arguments


def autodetect_scenarios(config):
    """
    Autodetect the scenarios in the config file
    """
    seir_modifiers_scenarios = None
    if config["seir_modifiers"].exists():
        if config["seir_modifiers"]["scenarios"].exists():
            seir_modifiers_scenarios = config["seir_modifiers"]["scenarios"].as_str_seq()

    outcome_modifiers_scenarios = None
    if config["outcomes"].exists() and config["outcome_modifiers"].exists():
        if config["outcome_modifiers"]["scenarios"].exists():
            outcome_modifiers_scenarios = config["outcome_modifiers"]["scenarios"].as_str_seq()

    outcome_modifiers_scenarios = as_list(outcome_modifiers_scenarios)
    seir_modifiers_scenarios = as_list(seir_modifiers_scenarios)

    if len(seir_modifiers_scenarios) != 1 or len(outcome_modifiers_scenarios) != 1:
        raise ValueError(
            f"Inference only support configurations files with one scenario, got"
            f"seir: {seir_modifiers_scenarios}"
            f"outcomes: {outcome_modifiers_scenarios}"
        )

    return seir_modifiers_scenarios[0], outcome_modifiers_scenarios[0]

# rewrite the get log loss functions as single functions, not in a class. This is not faster
# def get_logloss(proposal, inferpar, logloss, static_sim_arguments, modinf, silent=True, save=False):
#     if not inferpar.check_in_bound(proposal=proposal):
#         if not silent:
#             print("OUT OF BOUND!!")
#         return -np.inf, -np.inf, -np.inf
# 
#     snpi_df_mod, hnpi_df_mod = inferpar.inject_proposal(
#         proposal=proposal,
#         snpi_df=static_sim_arguments["snpi_df_ref"],
#         hnpi_df=static_sim_arguments["hnpi_df_ref"],
#     )
# 
#     ss = copy.deepcopy(static_sim_arguments)
#     ss["snpi_df_in"] = snpi_df_mod
#     ss["hnpi_df_in"] = hnpi_df_mod
#     del ss["snpi_df_ref"]
#     del ss["hnpi_df_ref"]
# 
#     outcomes_df = simulation_atomic(**ss, modinf=modinf, save=save)
# 
#     ll_total, logloss, regularizations = logloss.compute_logloss(
#         model_df=outcomes_df, subpop_names=modinf.subpop_struct.subpop_names
#     )
#     if not silent:
#         print(f"llik is {ll_total}")
# 
#     return ll_total, logloss, regularizations
# 
# def get_logloss_as_single_number(proposal, inferpar, logloss, static_sim_arguments, modinf, silent=True, save=False):
#     ll_total, logloss, regularizations = get_logloss(proposal, inferpar, logloss, static_sim_arguments, modinf, silent, save)
#     return ll_total



class GempyorInference:
    def __init__(
        self,
        config_filepath,
        run_id="test_run_id",
        prefix=None,
        first_sim_index=1,
        stoch_traj_flag=False,
        rng_seed=None,
        nslots=1,
        inference_filename_prefix="",  # usually for {global or chimeric}/{intermediate or final}
        inference_filepath_suffix="",  # usually for the slot_id
        out_run_id=None,  # if out_run_id is different from in_run_id, fill this
        out_prefix=None,  # if out_prefix is different from in_prefix, fill this
        path_prefix="",  # in case the data folder is on another directory
        autowrite_seir=False,
    ):
        # Config prep
        config.clear()
        config.read(user=False)
        config.set_file(path_prefix + config_filepath)

        self.seir_modifiers_scenario, self.outcome_modifiers_scenario = autodetect_scenarios(config)

        if run_id is None:
            run_id = file_paths.run_id()
        if prefix is None:
            prefix = config["name"].get() + "_inference_all" + "/" + run_id + "/"
        in_run_id = run_id
        if out_run_id is None:
            out_run_id = in_run_id
        in_prefix = prefix
        if out_prefix is None:
            out_prefix = in_prefix

        np.random.seed(rng_seed)

        write_csv = False
        write_parquet = True
        self.modinf = model_info.ModelInfo(
            config=config,
            nslots=nslots,
            seir_modifiers_scenario=self.seir_modifiers_scenario,
            outcome_modifiers_scenario=self.outcome_modifiers_scenario,
            write_csv=write_csv,
            write_parquet=write_parquet,
            first_sim_index=first_sim_index,
            in_run_id=in_run_id,
            in_prefix=in_prefix,
            inference_filename_prefix=inference_filename_prefix,
            inference_filepath_suffix=inference_filepath_suffix,
            out_run_id=out_run_id,
            out_prefix=out_prefix,
            stoch_traj_flag=stoch_traj_flag,
            config_filepath=config_filepath,
            path_prefix=path_prefix,
        )

        print(
            f"""  gempyor >> Running ***{'STOCHASTIC' if stoch_traj_flag else 'DETERMINISTIC'}*** simulation;\n"""
            f"""  gempyor >> ModelInfo {self.modinf.setup_name}; index: {self.modinf.first_sim_index}; run_id: {in_run_id},\n"""
            f"""  gempyor >> prefix: {self.modinf.in_prefix};"""  # ti: {s.ti}; tf: {s.tf};
        )

        self.already_built = False  # whether we have already built the costly objects that need just one build
        self.autowrite_seir = autowrite_seir

        ## Inference Stuff
        self.do_inference = False
        if config["inference"].exists():
            from . import inference_parameter, logloss

            if config["inference"]["method"].get("default") == "emcee":
                self.do_inference = True
                self.inference_method = "emcee"
                self.inferpar = inference_parameter.InferenceParameters(
                    global_config=config, subpop_names=self.modinf.subpop_struct.subpop_names
                )
                self.logloss = logloss.LogLoss(
                    inference_config=config["inference"],
                    path_prefix=path_prefix,
                    subpop_struct=self.modinf.subpop_struct,
                    time_setup=self.modinf.time_setup,
                )
                self.static_sim_arguments = get_static_arguments(self.modinf)

                print("Running Gempyor Inference")
                print(self.logloss)
                print(self.inferpar)

        self.silent = True
        self.save = False

    def set_silent(self, silent):
        self.silent = silent

    def set_save(self, save):
        self.save = save

    def get_all_sim_arguments(self):
        # inferpar, logloss, static_sim_arguments, modinf, proposal, silent, save
        return [self.inferpar, self.logloss, self.static_sim_arguments, self.modinf, self.silent, self.save]

    def get_logloss(self, proposal):
        if not self.inferpar.check_in_bound(proposal=proposal):
            if not self.silent:
                print("OUT OF BOUND!!")
            return -np.inf, -np.inf, -np.inf

        snpi_df_mod, hnpi_df_mod = self.inferpar.inject_proposal(
            proposal=proposal,
            snpi_df=self.static_sim_arguments["snpi_df_ref"],
            hnpi_df=self.static_sim_arguments["hnpi_df_ref"],
        )

        ss = copy.deepcopy(self.static_sim_arguments)
        ss["snpi_df_in"] = snpi_df_mod
        ss["hnpi_df_in"] = hnpi_df_mod
        del ss["snpi_df_ref"]
        del ss["hnpi_df_ref"]

        outcomes_df = simulation_atomic(**ss, modinf=self.modinf, save=self.save)

        ll_total, logloss, regularizations = self.logloss.compute_logloss(
            model_df=outcomes_df, subpop_names=self.modinf.subpop_struct.subpop_names
        )
        if not self.silent:
            print(f"llik is {ll_total}")

        return ll_total, logloss, regularizations

    def get_logloss_as_single_number(self, proposal):
        ll_total, logloss, regularizations = self.get_logloss(proposal)
        return ll_total

    def perform_test_run(self):
        ss = copy.deepcopy(self.static_sim_arguments)

        ss["snpi_df_in"] = ss["snpi_df_ref"]
        ss["hnpi_df_in"] = ss["hnpi_df_ref"]
        # delete the ref
        del ss["snpi_df_ref"]
        del ss["hnpi_df_ref"]

        hosp = simulation_atomic(**ss, modinf=self.modinf)

        ll_total, logloss, regularizations = self.logloss.compute_logloss(
            model_df=hosp, subpop_names=self.modinf.subpop_struct.subpop_names
        )
        print(
            f"test run successful 🎉, with logloss={ll_total:.1f} including {regularizations:.1f} for regularization ({regularizations/ll_total*100:.1f}%) "
        )
        return hosp, ll_total, logloss, regularizations

    def update_prefix(self, new_prefix, new_out_prefix=None):
        self.modinf.in_prefix = new_prefix
        if new_out_prefix is None:
            self.modinf.out_prefix = new_prefix
        else:
            self.modinf.out_prefix = new_out_prefix

    def update_run_id(self, new_run_id, new_out_run_id=None):
        self.modinf.in_run_id = new_run_id
        if new_out_run_id is None:
            self.modinf.out_run_id = new_run_id
        else:
            self.modinf.out_run_id = new_out_run_id

    def one_simulation_legacy(self, sim_id2write: int, load_ID: bool = False, sim_id2load: int = None):
        sim_id2write = int(sim_id2write)
        if load_ID:
            sim_id2load = int(sim_id2load)
        with Timer(f">>> GEMPYOR onesim {'(loading file)' if load_ID else '(from config)'}"):
            with Timer("onerun_SEIR"):
                seir.onerun_SEIR(
                    sim_id2write=sim_id2write,
                    modinf=self.modinf,
                    load_ID=load_ID,
                    sim_id2load=sim_id2load,
                    config=config,
                )
            if self.modinf.outcomes_config is not None:
                with Timer("onerun_OUTCOMES"):
                    outcomes.onerun_delayframe_outcomes(
                        sim_id2write=sim_id2write,
                        modinf=self.modinf,
                        load_ID=load_ID,
                        sim_id2load=sim_id2load,
                    )
        return 0

    def build_structure(self):
        (
            self.unique_strings,
            self.transition_array,
            self.proportion_array,
            self.proportion_info,
        ) = self.modinf.compartments.get_transition_array()
        self.already_built = True

    def write_last_seir(self, sim_id2write=None):
        if sim_id2write is None:
            sim_id2write = self.lastsim_sim_id2write
        out_df = seir.write_seir(sim_id2write, self.modinf, self.lastsim_states)
        return out_df

    # @profile()
    def one_simulation(
        self,
        sim_id2write: int,
        load_ID: bool = False,
        sim_id2load: int = None,
        parallel=False,
    ):
        sim_id2write = int(sim_id2write)
        self.lastsim_sim_id2write = sim_id2write
        self.lastsim_loadID = load_ID
        self.lastsim_sim_id2load = sim_id2load
        if load_ID:
            sim_id2load = int(sim_id2load)
            self.lastsim_sim_id2load = sim_id2load

        with Timer(f">>> GEMPYOR onesim {'(loading file)' if load_ID else '(from config)'}"):
            if not self.already_built and self.modinf.outcomes_config is not None:
                self.outcomes_parameters = outcomes.read_parameters_from_config(self.modinf)

            npi_outcomes = None
            if parallel:
                with Timer("//things"):
                    with ProcessPoolExecutor(max_workers=max(mp.cpu_count(), 3)) as executor:
                        ret_seir = executor.submit(seir.build_npi_SEIR, self.modinf, load_ID, sim_id2load, config)
                        if self.modinf.outcomes_config is not None and self.modinf.npi_config_outcomes:
                            ret_outcomes = executor.submit(
                                outcomes.build_outcome_modifiers,
                                self.modinf,
                                load_ID,
                                sim_id2load,
                                config,
                            )
                        if not self.already_built:
                            ret_comparments = executor.submit(self.modinf.compartments.get_transition_array)

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
                if self.modinf.outcomes_config is not None and self.modinf.npi_config_outcomes:
                    npi_outcomes = ret_outcomes.result()
            else:
                if not self.already_built:
                    self.build_structure()
                npi_seir = seir.build_npi_SEIR(
                    modinf=self.modinf, load_ID=load_ID, sim_id2load=sim_id2load, config=config
                )
                if self.modinf.outcomes_config is not None and self.modinf.npi_config_outcomes:
                    npi_outcomes = outcomes.build_outcome_modifiers(
                        modinf=self.modinf,
                        load_ID=load_ID,
                        sim_id2load=sim_id2load,
                        config=config,
                    )
            self.lastsim_npi_seir = npi_seir
            self.lastsim_npi_outcomes = npi_outcomes
            ### Run every time:
            with Timer("SEIR.parameters"):
                # Draw or load parameters

                p_draw = self.get_seir_parameters(load_ID=load_ID, sim_id2load=sim_id2load)
                # reduce them
                parameters = self.modinf.parameters.parameters_reduce(p_draw, npi_seir)
                # Parse them
                parsed_parameters = self.modinf.compartments.parse_parameters(
                    parameters, self.modinf.parameters.pnames, self.unique_strings
                )
                self.lastsim_p_draw = p_draw
                self.lastsim_parameters = parameters
                self.lastsim_parsed_parameters = parsed_parameters

            with Timer("onerun_SEIR.seeding"):
                if load_ID:
                    initial_conditions = self.modinf.initial_conditions.get_from_file(sim_id2load, modinf=self.modinf)
                    seeding_data, seeding_amounts = self.modinf.seeding.get_from_file(sim_id2load, modinf=self.modinf)
                else:
                    initial_conditions = self.modinf.initial_conditions.get_from_config(
                        sim_id2write, modinf=self.modinf
                    )
                    seeding_data, seeding_amounts = self.modinf.seeding.get_from_config(
                        sim_id2write, modinf=self.modinf
                    )
                self.lastsim_seeding_data = seeding_data
                self.lastsim_seeding_amounts = seeding_amounts
                self.lastsim_initial_conditions = initial_conditions

            with Timer("SEIR.compute"):
                states = seir.steps_SEIR(
                    self.modinf,
                    parsed_parameters,
                    self.transition_array,
                    self.proportion_array,
                    self.proportion_info,
                    initial_conditions,
                    seeding_data,
                    seeding_amounts,
                )
                self.lastsim_states = states

            with Timer("SEIR.postprocess"):
                if self.modinf.write_csv or self.modinf.write_parquet:
                    seir.write_spar_snpi(sim_id2write, self.modinf, p_draw, npi_seir)
                    if self.autowrite_seir:
                        out_df = seir.write_seir(sim_id2write, self.modinf, states)
                        self.lastsim_out_df = out_df

            loaded_values = None
            if load_ID:
                loaded_values = self.modinf.read_simID(ftype="hpar", sim_id=sim_id2load)
                self.lastsim_loaded_values = loaded_values

            # Compute outcomes
            if self.modinf.outcomes_config is not None:
                with Timer("onerun_delayframe_outcomes.compute"):
                    outcomes_df, hpar_df = outcomes.compute_all_multioutcomes(
                        modinf=self.modinf,
                        sim_id2write=sim_id2write,
                        parameters=self.outcomes_parameters,
                        loaded_values=loaded_values,
                        npi=npi_outcomes,
                    )
                    self.lastsim_outcomes_df = outcomes_df
                    self.lastsim_hpar_df = hpar_df

                with Timer("onerun_delayframe_outcomes.postprocess"):
                    outcomes.postprocess_and_write(
                        sim_id=sim_id2write,
                        modinf=self.modinf,
                        outcomes_df=outcomes_df,
                        hpar=hpar_df,
                        npi=npi_outcomes,
                    )
        return 0

    def plot_transition_graph(self, output_file="transition_graph", source_filters=[], destination_filters=[]):
        self.modinf.compartments.plot(
            output_file=output_file,
            source_filters=source_filters,
            destination_filters=destination_filters,
        )

    def get_outcome_npi(self, load_ID=False, sim_id2load=None, bypass_DF=None, bypass_FN=None):
        npi_outcomes = None
        if self.modinf.npi_config_outcomes:
            npi_outcomes = outcomes.build_outcome_modifiers(
                modinf=self.modinf,
                load_ID=load_ID,
                sim_id2load=sim_id2load,
                config=config,
                bypass_DF=bypass_DF,
                bypass_FN=bypass_FN,
            )
        return npi_outcomes

    def get_seir_npi(self, load_ID=False, sim_id2load=None, bypass_DF=None, bypass_FN=None):
        npi_seir = seir.build_npi_SEIR(
            modinf=self.modinf,
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
            param_df = self.modinf.read_simID(ftype="spar", sim_id=sim_id2load)

        if param_df is not None:
            p_draw = self.modinf.parameters.parameters_load(
                param_df=param_df,
                n_days=self.modinf.n_days,
                nsubpops=self.modinf.nsubpops,
            )
        else:
            p_draw = self.modinf.parameters.parameters_quick_draw(
                n_days=self.modinf.n_days, nsubpops=self.modinf.nsubpops
            )
        return p_draw

    def get_seir_parametersDF(self, load_ID=False, sim_id2load=None, bypass_DF=None, bypass_FN=None):
        p_draw = self.get_seir_parameters(
            load_ID=load_ID,
            sim_id2load=sim_id2load,
            bypass_DF=bypass_DF,
            bypass_FN=bypass_FN,
        )
        return self.modinf.parameters.getParameterDF(p_draw=p_draw)

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

        parameters = self.modinf.parameters.parameters_reduce(p_draw, npi_seir)

        full_df = pd.DataFrame()
        for i, subpop in enumerate(self.modinf.subpop_struct.subpop_names):
            a = pd.DataFrame(
                parameters[:, :, i].T,
                columns=self.modinf.parameters.pnames,
                index=pd.date_range(self.modinf.ti, self.modinf.tf, freq="D"),
            )
            a["subpop"] = subpop
            full_df = pd.concat([full_df, a])

        # for R, duplicate names are not allowed in index:
        full_df["date"] = full_df.index
        full_df = full_df.reset_index(drop=True)

        return full_df

    # TODO these function should support bypass
    def get_parsed_parameters_seir(
        self,
        load_ID=False,
        sim_id2load=None,
        # bypass_DF=None,
        # bypass_FN=None,
    ):
        if not self.already_built:
            self.build_structure()

        npi_seir = seir.build_npi_SEIR(modinf=self.modinf, load_ID=load_ID, sim_id2load=sim_id2load, config=config)
        p_draw = self.get_seir_parameters(load_ID=load_ID, sim_id2load=sim_id2load)

        parameters = self.modinf.parameters.parameters_reduce(p_draw, npi_seir)

        parsed_parameters = self.modinf.compartments.parse_parameters(
            parameters, self.modinf.parameters.pnames, self.unique_strings
        )
        return parsed_parameters

    def get_reduced_parameters_seir(
        self,
        load_ID=False,
        sim_id2load=None,
        # bypass_DF=None,
        # bypass_FN=None,
    ):
        npi_seir = seir.build_npi_SEIR(modinf=self.modinf, load_ID=load_ID, sim_id2load=sim_id2load, config=config)
        p_draw = self.get_seir_parameters(load_ID=load_ID, sim_id2load=sim_id2load)

        parameters = self.modinf.parameters.parameters_reduce(p_draw, npi_seir)

        parsed_parameters = self.modinf.compartments.parse_parameters(
            parameters, self.modinf.parameters.pnames, self.unique_strings
        )
        return parsed_parameters


def paramred_parallel(run_spec, snpi_fn):
    config_filepath = run_spec["config"]
    gempyor_inference = GempyorInference(
        config_filepath=config_filepath,
        run_id="test_run_id",
        prefix="test_prefix/",
        first_sim_index=1,
        seir_modifiers_scenario="inference",  # NPIs scenario to use
        outcome_modifiers_scenario="med",  # Outcome scenario to use
        stoch_traj_flag=False,
        path_prefix=run_spec["geodata"],  # prefix where to find the folder indicated in subpop_setup$
    )

    snpi = pq.read_table(snpi_fn).to_pandas()

    npi_seir = gempyor_inference.get_seir_npi(bypass_DF=snpi)

    # params_draw_df = gempyor_inference.get_seir_parametersDF()  # could also accept (load_ID=True, sim_id2load=XXX) or (bypass_DF=<some_spar_df>) or (bypass_FN=<some_spar_filename>)
    params_draw_arr = gempyor_inference.get_seir_parameters(
        bypass_FN=snpi_fn.replace("snpi", "spar")
    )  # could also accept (load_ID=True, sim_id2load=XXX) or (bypass_DF=<some_spar_df>) or (bypass_FN=<some_spar_filename>)
    param_reduc_from = gempyor_inference.get_seir_parameter_reduced(npi_seir=npi_seir, p_draw=params_draw_arr)

    return param_reduc_from


def paramred_parallel_config(run_spec, dummy):
    config_filepath = run_spec["config"]
    gempyor_inference = GempyorInference(
        config_filepath=config_filepath,
        run_id="test_run_id",
        prefix="test_prefix/",
        first_sim_index=1,
        seir_modifiers_scenario="inference",  # NPIs scenario to use
        outcome_modifiers_scenario="med",  # Outcome scenario to use
        stoch_traj_flag=False,
        path_prefix=run_spec["geodata"],  # prefix where to find the folder indicated in subpop_setup$
    )

    npi_seir = gempyor_inference.get_seir_npi()

    params_draw_arr = (
        gempyor_inference.get_seir_parameters()
    )  # could also accept (load_ID=True, sim_id2load=XXX) or (bypass_DF=<some_spar_df>) or (bypass_FN=<some_spar_filename>)
    param_reduc_from = gempyor_inference.get_seir_parameter_reduced(npi_seir=npi_seir, p_draw=params_draw_arr)

    return param_reduc_from
