import itertools
import time
import numpy as np
import pandas as pd
import scipy
import tqdm.contrib.concurrent
import xarray as xr

from . import NPI, model_info, steps_rk4
from .utils import Timer, print_disk_diagnosis, read_df
import logging

logger = logging.getLogger(__name__)


def build_step_source_arg(
    modinf,
    parsed_parameters,
    transition_array,
    proportion_array,
    proportion_info,
    initial_conditions,
    seeding_data,
    seeding_amounts,
):
    if "integration" in modinf.seir_config.keys():
        if "method" in modinf.seir_config["integration"].keys():
            integration_method = modinf.seir_config["integration"]["method"].get()
            if integration_method == "best.current":
                integration_method = "rk4.jit"
            if integration_method == "rk4":
                integration_method = "rk4.jit"
            if integration_method not in ["rk4.jit", "legacy"]:
                raise ValueError(f"Unknown integration method {integration_method}.")
        if "dt" in modinf.seir_config["integration"].keys():
            dt = float(
                eval(str(modinf.seir_config["integration"]["dt"].get()))
            )  # ugly way to parse string and formulas
        else:
            dt = 2.0
    else:
        integration_method = "rk4.jit"
        dt = 2.0
        logging.info(
            f"Integration method not provided, assuming type {integration_method} with dt=2"
        )

    ## The type is very important for the call to the compiled function, and e.g mixing an int64 for an int32 can
    ## result in serious error. Note that "In Microsoft C, even on a 64 bit system, the size of the long int data type
    ## is 32 bits." so upstream user need to specifcally cast everything to int64
    ## Somehow only mobility data is caseted by this function, but perhaps we should handle it all here ?
    assert type(modinf.mobility) == scipy.sparse.csr_matrix
    mobility_data = modinf.mobility.data
    mobility_data = mobility_data.astype("float64")
    assert type(modinf.compartments.compartments.shape[0]) == int
    assert type(modinf.nsubpops) == int
    assert modinf.n_days > 1
    assert parsed_parameters.shape[1:3] == (modinf.n_days, modinf.nsubpops)
    assert type(dt) == float
    assert type(transition_array[0][0]) == np.int64
    assert type(proportion_array[0]) == np.int64
    assert type(proportion_info[0][0]) == np.int64
    assert initial_conditions.shape == (
        modinf.compartments.compartments.shape[0],
        modinf.nsubpops,
    )
    assert type(initial_conditions[0][0]) == np.float64
    # Test of empty seeding:
    assert len(seeding_data.keys()) == 4

    keys_ref = [
        "seeding_sources",
        "seeding_destinations",
        "seeding_subpops",
        "day_start_idx",
    ]
    for key, item in seeding_data.items():
        assert key in keys_ref
        if key == "day_start_idx":
            assert len(item) == modinf.n_days + 1
            # assert (item == np.zeros(s.n_days + 1, dtype=np.int64)).all()
        # else:
        #     assert item.size == np.array([], dtype=np.int64)
        assert item.dtype == np.int64

    if len(mobility_data) > 0:
        assert type(mobility_data[0]) == np.float64
        assert len(mobility_data) == len(modinf.mobility.indices)
        assert type(modinf.mobility.indices[0]) == np.int32
        assert len(modinf.mobility.indptr) == modinf.nsubpops + 1
        assert type(modinf.mobility.indptr[0]) == np.int32

    assert len(modinf.subpop_pop) == modinf.nsubpops
    assert type(modinf.subpop_pop[0]) == np.int64

    assert dt <= 1.0 or dt == 2.0

    fnct_args = {
        "ncompartments": modinf.compartments.compartments.shape[0],
        "nspatial_nodes": modinf.nsubpops,
        "ndays": modinf.n_days,
        "parameters": parsed_parameters,
        "dt": dt,
        "integration_method": integration_method,
        "transitions": transition_array,
        "proportion_info": proportion_info,
        "transition_sum_compartments": proportion_array,
        "initial_conditions": initial_conditions,
        "seeding_data": seeding_data,
        "seeding_amounts": seeding_amounts,
        "mobility_data": mobility_data,
        "mobility_row_indices": modinf.mobility.indices,
        "mobility_data_indices": modinf.mobility.indptr,
        "population": modinf.subpop_pop,
        "stochastic_p": modinf.stoch_traj_flag,
    }
    return fnct_args


def steps_SEIR(
    modinf,
    parsed_parameters,
    transition_array,
    proportion_array,
    proportion_info,
    initial_conditions,
    seeding_data,
    seeding_amounts,
):
    fnct_args = build_step_source_arg(
        modinf,
        parsed_parameters,
        transition_array,
        proportion_array,
        proportion_info,
        initial_conditions,
        seeding_data,
        seeding_amounts,
    )

    integration_method = fnct_args["integration_method"]
    fnct_args.pop("integration_method")

    logging.debug(f"Integrating with method {integration_method}")

    if integration_method == "legacy":
        seir_sim = seir_sim = steps_rk4.rk4_integration(**fnct_args, method="legacy")
    elif integration_method == "rk4.jit":
        if modinf.stoch_traj_flag == True:
            raise ValueError(
                f"with method {integration_method}, only deterministic "
                f"integration is possible (got stoch_straj_flag={modinf.stoch_traj_flag}"
            )
        seir_sim = steps_rk4.rk4_integration(**fnct_args, silent=True)
    else:
        from .dev import steps as steps_experimental

        logging.critical(
            "Experimental !!! These methods are not ready for production ! "
        )
        if integration_method in [
            "scipy.solve_ivp",
            "scipy.odeint",
            "scipy.solve_ivp2",
            "scipy.odeint2",
        ]:
            if modinf.stoch_traj_flag == True:
                raise ValueError(
                    f"with method {integration_method}, only deterministic "
                    f"integration is possible (got stoch_straj_flag={modinf.stoch_traj_flag}"
                )
            seir_sim = steps_experimental.ode_integration(
                **fnct_args, integration_method=integration_method
            )
        elif integration_method == "rk4.jit1":
            seir_sim = steps_experimental.rk4_integration1(**fnct_args)
        elif integration_method == "rk4.jit2":
            seir_sim = steps_experimental.rk4_integration2(**fnct_args)
        elif integration_method == "rk4.jit3":
            seir_sim = steps_experimental.rk4_integration3(**fnct_args)
        elif integration_method == "rk4.jit4":
            seir_sim = steps_experimental.rk4_integration4(**fnct_args)
        elif integration_method == "rk4.jit5":
            seir_sim = steps_experimental.rk4_integration5(**fnct_args)
        elif integration_method == "rk4.jit6":
            seir_sim = steps_experimental.rk4_integration6(**fnct_args)
        elif integration_method == "rk4.jit.smart":
            seir_sim = steps_experimental.rk4_integration2_smart(**fnct_args)
        elif integration_method == "rk4_aot":
            seir_sim = steps_experimental.rk4_aot(**fnct_args)
        else:
            raise ValueError(f"Unknow integration scheme, got {integration_method}")

    # We return an xarray instead of a ndarray now
    compartment_coords = {}
    compartment_df = modinf.compartments.get_compartments_explicitDF()
    # Iterate over columns of the DataFrame and populate the dictionary
    for column in compartment_df.columns:
        compartment_coords[column] = ("compartment", compartment_df[column].tolist())

    # comparment is a dimension with coordinate from each of the compartments
    states = xr.Dataset(
        data_vars=dict(
            prevalence=(["date", "compartment", "subpop"], seir_sim[0]),
            incidence=(["date", "compartment", "subpop"], seir_sim[1]),
        ),
        coords=dict(
            date=pd.date_range(modinf.ti, modinf.tf, freq="D"),
            **compartment_coords,
            subpop=modinf.subpop_struct.subpop_names,
        ),
        attrs=dict(
            description="Dynamical simulation results", run_id=modinf.in_run_id
        ),  # TODO add more information
    )

    return states


def build_npi_SEIR(
    modinf, load_ID, sim_id2load, config, bypass_DF=None, bypass_FN=None
):
    with Timer("SEIR.NPI"):
        loaded_df = None
        if bypass_DF is not None:
            loaded_df = bypass_DF
        elif bypass_FN is not None:
            loaded_df = read_df(fname=bypass_FN)
        elif load_ID == True:
            loaded_df = modinf.read_simID(ftype="snpi", sim_id=sim_id2load)

        if loaded_df is not None:
            npi = NPI.NPIBase.execute(
                npi_config=modinf.npi_config_seir,
                modinf=modinf,
                modifiers_library=modinf.seir_modifiers_library,
                subpops=modinf.subpop_struct.subpop_names,
                loaded_df=loaded_df,
                pnames_overlap_operation_sum=modinf.parameters.stacked_modifier_method[
                    "sum"
                ],
                pnames_overlap_operation_reductionprod=modinf.parameters.stacked_modifier_method[
                    "reduction_product"
                ],
            )
        else:
            npi = NPI.NPIBase.execute(
                npi_config=modinf.npi_config_seir,
                modinf=modinf,
                modifiers_library=modinf.seir_modifiers_library,
                subpops=modinf.subpop_struct.subpop_names,
                pnames_overlap_operation_sum=modinf.parameters.stacked_modifier_method[
                    "sum"
                ],
                pnames_overlap_operation_reductionprod=modinf.parameters.stacked_modifier_method[
                    "reduction_product"
                ],
            )
    return npi


def onerun_SEIR(
    sim_id2write: int,
    modinf: model_info.ModelInfo,
    load_ID: bool = False,
    sim_id2load: int = None,
    config=None,
):
    np.random.seed()
    npi = None
    if modinf.npi_config_seir:
        npi = build_npi_SEIR(
            modinf=modinf, load_ID=load_ID, sim_id2load=sim_id2load, config=config
        )

    with Timer("onerun_SEIR.compartments"):
        (
            unique_strings,
            transition_array,
            proportion_array,
            proportion_info,
        ) = modinf.compartments.get_transition_array()

    with Timer("onerun_SEIR.seeding"):
        if load_ID:
            initial_conditions = modinf.initial_conditions.get_from_file(
                sim_id2load, modinf=modinf
            )
            seeding_data, seeding_amounts = modinf.seeding.get_from_file(
                sim_id2load, modinf=modinf
            )
        else:
            initial_conditions = modinf.initial_conditions.get_from_config(
                sim_id2write, modinf=modinf
            )
            seeding_data, seeding_amounts = modinf.seeding.get_from_config(
                sim_id2write, modinf=modinf
            )

    with Timer("onerun_SEIR.parameters"):
        # Draw or load parameters
        if load_ID:
            p_draw = modinf.parameters.parameters_load(
                param_df=modinf.read_simID(ftype="spar", sim_id=sim_id2load),
                n_days=modinf.n_days,
                nsubpops=modinf.nsubpops,
            )
        else:
            p_draw = modinf.parameters.parameters_quick_draw(
                n_days=modinf.n_days, nsubpops=modinf.nsubpops
            )
        # reduce them
        parameters = modinf.parameters.parameters_reduce(p_draw, npi)
        log_debug_parameters(p_draw, "Parameters without seir_modifiers")
        log_debug_parameters(parameters, "Parameters with seir_modifiers")

        # Parse them
        parsed_parameters = modinf.compartments.parse_parameters(
            parameters, modinf.parameters.pnames, unique_strings
        )
        log_debug_parameters(parsed_parameters, "Unique Parameters used by transitions")

    with Timer("onerun_SEIR.compute"):
        states = steps_SEIR(
            modinf,
            parsed_parameters,
            transition_array,
            proportion_array,
            proportion_info,
            initial_conditions,
            seeding_data,
            seeding_amounts,
        )

    with Timer("onerun_SEIR.postprocess"):
        if modinf.write_csv or modinf.write_parquet:
            write_spar_snpi(sim_id2write, modinf, p_draw, npi)
            out_df = write_seir(sim_id2write, modinf, states)
    return out_df


def run_parallel_SEIR(modinf, config, *, n_jobs=1):
    start = time.monotonic()
    sim_ids = np.arange(1, modinf.nslots + 1)

    if n_jobs == 1:  # run single process for debugging/profiling purposes
        for sim_id in tqdm.tqdm(sim_ids):
            onerun_SEIR(
                sim_id2write=sim_id,
                modinf=modinf,
                load_ID=False,
                sim_id2load=None,
                config=config,
            )
    else:
        tqdm.contrib.concurrent.process_map(
            onerun_SEIR,
            sim_ids,
            itertools.repeat(modinf),
            itertools.repeat(False),
            itertools.repeat(None),
            itertools.repeat(config),
            max_workers=n_jobs,
        )

    logging.info(
        f""">> {modinf.nslots} seir simulations completed in {time.monotonic() - start:.1f} seconds"""
    )


def states2Df(modinf, states):
    # Tidyup data for  R, to save it:
    #
    # Write output to .snpi.*, .spar.*, and .seir.* files
    # states is  # both are [ndays x ncompartments x nspatial_nodes ] -> this is important here

    # add line of zero to diff, so we get the real cumulative.
    # states_diff = np.zeros((states_cumu.shape[0] + 1, *states_cumu.shape[1:]))
    # states_diff[1:, :, :] = states_cumu
    # states_diff = np.diff(states_diff, axis=0)

    ts_index = pd.MultiIndex.from_product(
        [
            pd.date_range(modinf.ti, modinf.tf, freq="D"),
            modinf.compartments.compartments["name"],
        ],
        names=["date", "mc_name"],
    )
    # prevalence data, we use multi.index dataframe, sparring us the array manipulation we use to do
    prev_df = pd.DataFrame(
        data=states["prevalence"]
        .to_numpy()
        .reshape(modinf.n_days * modinf.compartments.get_ncomp(), modinf.nsubpops),
        index=ts_index,
        columns=modinf.subpop_struct.subpop_names,
    ).reset_index()
    prev_df = pd.merge(
        left=modinf.compartments.get_compartments_explicitDF(),
        right=prev_df,
        how="right",
        on="mc_name",
    )
    prev_df.insert(loc=0, column="mc_value_type", value="prevalence")

    ts_index = pd.MultiIndex.from_product(
        [
            pd.date_range(modinf.ti, modinf.tf, freq="D"),
            modinf.compartments.compartments["name"],
        ],
        names=["date", "mc_name"],
    )

    incid_df = pd.DataFrame(
        data=states["incidence"]
        .to_numpy()
        .reshape(modinf.n_days * modinf.compartments.get_ncomp(), modinf.nsubpops),
        index=ts_index,
        columns=modinf.subpop_struct.subpop_names,
    ).reset_index()
    incid_df = pd.merge(
        left=modinf.compartments.get_compartments_explicitDF(),
        right=incid_df,
        how="right",
        on="mc_name",
    )
    incid_df.insert(loc=0, column="mc_value_type", value="incidence")

    out_df = pd.concat((incid_df, prev_df), axis=0).set_index("date")

    out_df["date"] = out_df.index

    return out_df


def write_spar_snpi(sim_id, modinf, p_draw, npi):
    # NPIs
    if npi is not None:
        modinf.write_simID(ftype="snpi", sim_id=sim_id, df=npi.getReductionDF())
    # Parameters
    modinf.write_simID(
        ftype="spar", sim_id=sim_id, df=modinf.parameters.getParameterDF(p_draw=p_draw)
    )


def write_seir(sim_id, modinf, states):
    # print_disk_diagnosis()
    out_df = states2Df(modinf, states)
    modinf.write_simID(ftype="seir", sim_id=sim_id, df=out_df)

    return out_df


def log_debug_parameters(params, prefix):
    if logging.getLogger().isEnabledFor(logging.DEBUG):
        logging.debug(prefix)
        for parameter in params:
            try:
                logging.debug(
                    f"""    shape {parameter.shape}, type {parameter.dtype}, range [{parameter.min()}, {parameter.mean()}, {parameter.max()}]"""
                )
            except:
                logging.debug(f"""    value {parameter}""")
