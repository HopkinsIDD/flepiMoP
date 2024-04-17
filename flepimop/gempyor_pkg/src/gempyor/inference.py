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

# TODO: there is way to many of these functions, merge with the R interface.py implementation to avoid code duplication
def simulation_atomic(*, snpi_df_in, hnpi_df_in, modinf, p_draw, unique_strings, transition_array, proportion_array, proportion_info, initial_conditions, seeding_data, seeding_amounts,outcomes_parameters, save=False):
    
    # We need to reseed because subprocess inherit of the same random generator state.
    np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))
    random_id = np.random.randint(0,1e8)

    npi_seir = seir.build_npi_SEIR(
        modinf=modinf, load_ID=False, sim_id2load=None, config=config, 
        bypass_DF=snpi_df_in
    )

    if modinf.npi_config_outcomes:
        npi_outcomes = outcomes.build_outcome_modifiers(
                    modinf=modinf,
                    load_ID=False,
                    sim_id2load=None,
                    config=config,
                    bypass_DF=hnpi_df_in
                )
    else:
        npi_outcomes = None

    # reduce them
    parameters = modinf.parameters.parameters_reduce(p_draw, npi_seir)
    # Parse them
    parsed_parameters = modinf.compartments.parse_parameters(
        parameters, modinf.parameters.pnames, unique_strings
    )

    # Convert the seeding data dictionnary to a numba dictionnary
    seeding_data_nbdict = nb.typed.Dict.empty(
        key_type=nb.types.unicode_type,
        value_type=nb.types.int64[:])

    for k,v in seeding_data.items():
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
        bypass_seir_xr=states
    )
    outcomes_df, hpar, hnpi = outcomes.postprocess_and_write(
        sim_id=0,
        modinf=modinf,
        outcomes_df=outcomes_df,
        hpar=hpar_df,
        npi=npi_outcomes,
    )
    
    if save:
        modinf.write_simID(ftype="hosp", sim_id=random_id, df=outcomes_df)
    # needs to be after write... because parquet write discard the index.
    outcomes_df = outcomes_df.set_index("date") # after writing

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
    npi_seir = seir.build_npi_SEIR(
        modinf=modinf, load_ID=False, sim_id2load=None, config=config
    )
    if modinf.npi_config_outcomes:
        npi_outcomes = outcomes.build_outcome_modifiers(
                    modinf=modinf,
                    load_ID=False,
                    sim_id2load=None,
                    config=config,
                )
    else:
        npi_outcomes = None

    p_draw = modinf.parameters.parameters_quick_draw(
                    n_days=modinf.n_days, nsubpops=modinf.nsubpops
                )

    initial_conditions = modinf.initial_conditions.get_from_config(sim_id=0, setup=modinf)
    seeding_data, seeding_amounts = modinf.seeding.get_from_config(sim_id=0, setup=modinf)

    # reduce them
    parameters = modinf.parameters.parameters_reduce(p_draw, npi_seir)
            # Parse them
    parsed_parameters = modinf.compartments.parse_parameters(
        parameters, modinf.parameters.pnames, unique_strings
    )

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

        coords=dict(
                    date=pd.date_range(modinf.ti, modinf.tf, freq="D"),
                    **compartment_coords,
                    subpop=modinf.subpop_struct.subpop_names
                )

        zeros = np.zeros((len(coords["date"]), len(coords["mc_name"][1]), len(coords["subpop"])))
        states = xr.Dataset(
                data_vars=dict(
                    prevalence=(["date", "compartment", "subpop"],    zeros),
                    incidence=(["date", "compartment", "subpop"],     zeros),
                ),
                coords=coords,
                attrs=dict(description="Dynamical simulation results", run_id=modinf.in_run_id) # TODO add more information
            )

    snpi_df_ref = npi_seir.getReductionDF()

    outcomes_df, hpar_df = outcomes.compute_all_multioutcomes(
        modinf=modinf,
        sim_id2write=0,
        parameters=outcomes_parameters,
        loaded_values=None,
        npi=npi_outcomes,
        bypass_seir_xr=states
    )
    outcomes_df_ref, hpar_ref, hnpi_df_ref = outcomes.postprocess_and_write(
        sim_id=0,
        modinf=modinf,
        outcomes_df=outcomes_df,
        hpar=hpar_df,
        npi=npi_outcomes,
    )
    outcomes_df_ref["time"] = outcomes_df_ref["date"] #which one should it be ?
    modinf.write_simID(ftype="hosp", sim_id=0, df=outcomes_df_ref)
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



