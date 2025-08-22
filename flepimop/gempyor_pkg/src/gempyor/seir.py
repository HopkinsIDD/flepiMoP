"""
Abstractions for interacting with the SEIR compartments of the model.
"""

import itertools
import logging
import time

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy
import tqdm.contrib.concurrent
import xarray as xr

from . import NPI, steps_rk4
from .model_info import ModelInfo
from .utils import Timer, _nslots_random_seeds, read_df

logger = logging.getLogger(__name__)


def check_parameter_positivity(
    parsed_parameters: npt.NDArray[np.float64],
    parameter_names: list[str],
    dates: pd.DatetimeIndex,
    subpop_names: list[str],
) -> None:
    """
    Identify earliest negative values for parameters after modifiers have been applied.

    Args:
        parsed_parameters: An array of parameter values with shape
            (n_parameters, n_days, n_subpops).
        parameter_names: A list of parameter names which correspond to the first
            dimension of `parsed_parameters`.
        dates: A pandas DatetimeIndex representing the dates of the parameters and
            correspond to the second dimension of `parsed_parameters`.
        subpop_names: A list of subpopulation names which correspond to the third
            dimension of `parsed_parameters`.

    Raises:
        ValueError: If `parsed_parameters` contains negative values.

    Returns:
        None

    Examples:
        >>> import numpy as np
        >>> import pandas as pd
        >>> from gempyor.seir import check_parameter_positivity
        >>> parsed_parameters = np.ones((3, 5, 2))
        >>> parameter_names = ["param1", "param2", "param3"]
        >>> dates = pd.date_range("2023-01-01", periods=5)
        >>> subpop_names = ["subpop1", "subpop2"]
        >>> check_parameter_positivity(
        ...     parsed_parameters, parameter_names, dates, subpop_names
        ... ) is None
        True
        >>> parsed_parameters[1, 1, 1] = -1
        >>> check_parameter_positivity(
        ...     parsed_parameters, parameter_names, dates, subpop_names
        ... )
        Traceback (most recent call last):
            ...
        ValueError: There are negative parameter errors in subpops subpop2, starting from date 2023-01-02 in parameters param2.
        >>> parsed_parameters[2, 2, 1] = -1
        >>> check_parameter_positivity(
        ...     parsed_parameters, parameter_names, dates, subpop_names
        ... )
        Traceback (most recent call last):
            ...
        ValueError: There are negative parameter errors in subpops subpop2, starting from date 2023-01-02 in parameters param2, param3.
    """
    if len(negative_index_parameters := np.argwhere(parsed_parameters < 0)) > 0:
        neg_params = set()
        neg_subpops = set()
        first_neg_date_j = len(dates) + 1
        for i, j, k in negative_index_parameters:
            neg_params.add(parameter_names[i])
            neg_subpops.add(subpop_names[k])
            first_neg_date_j = min(first_neg_date_j, j)
        first_neg_date = dates[first_neg_date_j].date()
        raise ValueError(
            "There are negative parameter errors in subpops "
            f"{', '.join(sorted(neg_subpops))}, starting from "
            f"date {first_neg_date} in parameters "
            f"{', '.join(sorted(neg_params))}."
        )


def build_step_source_arg(
    modinf: ModelInfo,
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
            integration_method = modinf.seir_config["integration"]["method"].as_str()
            if integration_method == "best.current":
                integration_method = "rk4.jit"
            if integration_method == "rk4":
                integration_method = "rk4.jit"
            if integration_method not in ["rk4.jit", "euler", "stochastic"]:
                raise ValueError(
                    f"Unknown integration method given, '{integration_method}'."
                )
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
            f"Integration method not provided, assuming type {integration_method} with dt={dt}"
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
    }

    check_parameter_positivity(
        fnct_args["parameters"],
        modinf.parameters.pnames,
        modinf.dates,
        modinf.subpop_struct.subpop_names,
    )

    return fnct_args


def steps_SEIR(
    modinf: ModelInfo,
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

    if integration_method == "euler":
        seir_sim = steps_rk4.rk4_integration(**fnct_args, method="euler")
    elif integration_method == "stochastic":
        seir_sim = steps_rk4.rk4_integration(**fnct_args, method="stochastic")
    elif integration_method == "rk4.jit":
        seir_sim = steps_rk4.rk4_integration(**fnct_args, silent=True)
    else:
        if integration_method in {
            "scipy.solve_ivp",
            "scipy.odeint",
            "scipy.solve_ivp2",
            "scipy.odeint2",
            "rk4.jit1",
            "rk4.jit2",
            "rk4.jit3",
            "rk4.jit4",
            "rk4.jit5",
            "rk4.jit6",
            "rk4.jit.smart",
            "rk4_aot",
        }:
            logger.critical(
                "The '%s' integration method is considered experimental, please use the 'rk4_experimental' git branch.",
                integration_method,
            )
        raise ValueError(f"Unknown integration method given, '{integration_method}'.")

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
    modinf: ModelInfo,
    load_ID,
    sim_id2load,
    config,
    bypass_DF=None,
    bypass_FN=None,
):
    with Timer("SEIR.NPI"):
        loaded_df = None
        if bypass_DF is not None:
            loaded_df = bypass_DF
        elif bypass_FN is not None:
            loaded_df = read_df(fname=bypass_FN)
        elif load_ID == True:
            loaded_df = modinf.read_simID(ftype="snpi", sim_id=sim_id2load)

        npi = NPI.NPIBase.execute(
            npi_config=modinf.npi_config_seir,
            modinf_ti=modinf.ti,
            modinf_tf=modinf.tf,
            modifiers_library=modinf.seir_modifiers_library,
            subpops=modinf.subpop_struct.subpop_names,
            loaded_df=loaded_df,
            pnames_overlap_operation_sum=modinf.parameters.stacked_modifier_method["sum"],
            pnames_overlap_operation_reductionprod=modinf.parameters.stacked_modifier_method[
                "reduction_product"
            ],
        )
    return npi


def onerun_SEIR(
    sim_id2write: int,
    modinf: ModelInfo,
    load_ID: bool = False,
    sim_id2load: int = None,
    config=None,
) -> pd.DataFrame:
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
        else:
            initial_conditions = modinf.initial_conditions.get_from_config(
                sim_id2write, modinf=modinf
            )
        seeding_data, seeding_amounts = modinf.get_seeding_data(
            sim_id=sim_id2load if load_ID else sim_id2write
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
        out_df = states2Df(modinf, states)
        if modinf.write_csv or modinf.write_parquet:
            write_spar_snpi(sim_id2write, modinf, p_draw, npi)
            write_seir(sim_id2write, modinf, out_df)
        return out_df


def _onerun_SEIR_with_random_seed(
    random_seed: int,
    sim_id2write: int,
    modinf: ModelInfo,
    load_ID: bool = False,
    sim_id2load: int = None,
    config=None,
) -> pd.DataFrame:
    """
    Wrapper function to `onerun_SEIR` that sets a random seed.

    Args:
        random_seed: The random seed to use.
        sim_id2write: The simulation ID to write.
        modinf: The ModelInfo object.
        load_ID: Whether to load the simulation ID.
        sim_id2load: The simulation ID to load.
        config: The configuration.

    Returns:
        A pandas DataFrame containing the simulated SEIR output.

    See Also:
        `onerun_SEIR`

    """
    np.random.seed(seed=random_seed)
    modinf.parameters.reinitialize_distributions()
    return onerun_SEIR(
        sim_id2write, modinf, load_ID=load_ID, sim_id2load=sim_id2load, config=config
    )


def run_parallel_SEIR(modinf: ModelInfo, config, *, n_jobs=1) -> None:
    """
    Run SEIR simulations in parallel.

    Args:
        modinf: The ModelInfo object.
        config: The configuration.
        n_jobs: The number of parallel jobs to run. Default is 1 (no parallelization).

    Notes:
        Successive calls to this function will produce different samples for random
        parameters.
    """
    start = time.monotonic()
    sim_ids = np.arange(1, modinf.nslots + 1)
    random_seeds = _nslots_random_seeds(modinf.nslots)
    if n_jobs == 1:  # run single process for debugging/profiling purposes
        for sim_id in tqdm.tqdm(sim_ids):
            _onerun_SEIR_with_random_seed(
                random_seeds[sim_id - 1],
                sim_id,
                modinf,
                load_ID=False,
                sim_id2load=None,
                config=config,
            )
    else:
        tqdm.contrib.concurrent.process_map(
            _onerun_SEIR_with_random_seed,
            random_seeds,
            sim_ids,
            itertools.repeat(modinf),
            itertools.repeat(False),
            itertools.repeat(None),
            itertools.repeat(config),
            max_workers=n_jobs,
        )

    logging.info(
        f">> {modinf.nslots} seir simulations completed "
        f"in {time.monotonic() - start:.1f} seconds"
    )


def states2Df(modinf: ModelInfo, states):
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


def write_spar_snpi(sim_id: int, modinf: ModelInfo, p_draw, npi):
    # NPIs
    if npi is not None:
        modinf.write_simID(ftype="snpi", sim_id=sim_id, df=npi.getReductionDF())
    # Parameters
    modinf.write_simID(
        ftype="spar", sim_id=sim_id, df=modinf.parameters.getParameterDF(p_draw=p_draw)
    )


def write_seir(sim_id, modinf: ModelInfo, out_df):
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
