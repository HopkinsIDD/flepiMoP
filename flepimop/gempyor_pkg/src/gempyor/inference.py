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

# TODO: should be able to draw e.g from an initial condition folder buuut keep the draw as a blob
# so it is saved by emcee, so I can build a posterio

class Inference():
    def __init__(self, inference_config: confuse.ConfigView, modinf) -> None:
        inferpar = inference_parameter.InferenceParameters(global_config=config, modinf=modinf)

        

    def test_run(self, modinf):
        ss = copy.deepcopy(self.static_sim_arguments)
        ss["snpi_df_in"] = ss["snpi_df_ref"]
        ss["hnpi_df_in"] = ss["hnpi_df_ref"]
        # delete the ref
        del ss["snpi_df_ref"]
        del ss["hnpi_df_ref"]

        hosp = simulation_atomic(**ss, modinf=modinf)

        ll_total, logloss, regularizations = loss.compute_logloss(model_df=hosp, modinf=modinf)
        print(f"test run successful 🎉, with logloss={ll_total:.1f} including {regularizations:.1f} for regularization ({regularizations/ll_total*100:.1f}%) ")



def emcee_logprob(proposal, modinf, inferpar, loss, static_sim_arguments, save=False, silent=True):
    if not inferpar.check_in_bound(proposal=proposal):
        if not silent:
            print("OUT OF BOUND!!")
        return -np.inf

    snpi_df_mod, hnpi_df_mod = inferpar.inject_proposal(
        proposal=proposal, snpi_df=static_sim_arguments["snpi_df_ref"], hnpi_df=static_sim_arguments["hnpi_df_ref"]
    )

    ss = copy.deepcopy(static_sim_arguments)
    ss["snpi_df_in"] = snpi_df_mod
    ss["hnpi_df_in"] = hnpi_df_mod
    del ss["snpi_df_ref"]
    del ss["hnpi_df_ref"]

    outcomes_df = simulation_atomic(**ss, modinf=modinf, save=save)

    ll_total, logloss, regularizations = loss.compute_logloss(model_df=outcomes_df, modinf=modinf)
    if not silent:
        print(f"llik is {ll_total}")

    return ll_total


# TODO: there is way to many of these functions, merge with the R interface.py implementation to avoid code duplication
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
        sim_id=0,
        modinf=modinf,
        outcomes_df=outcomes_df,
        hpar=hpar_df,
        npi=npi_outcomes,
    )

    if save:
        modinf.write_simID(ftype="hosp", sim_id=random_id, df=outcomes_df)
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
    )
    outcomes_df_ref["time"] = outcomes_df_ref["date"]  # which one should it be ?
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


def find_walkers_to_sample(inferpar, sampler_output, nsamples, nwalker, nthin):
    # Find the good walkers
    if nsamples < nwalker:
        pass
        # easy case: sample the best llik (could also do random)
    
    last_llik = sampler_output.get_log_prob()[-1,:]
    sampled_slots = last_llik > (last_llik.mean()-1*last_llik.std())
    print(f"there are {sampled_slots.sum()}/{len(sampled_slots)} good walkers... keeping these")
    # TODO this function give back 



    good_samples =  sampler.get_chain()[:,sampled_slots,:]



    step_number = -1
    exported_samples = np.empty((nsamples,inferpar.get_dim()))
    for i in range(nsamples):
        exported_samples[i,:] = good_samples[step_number - thin*(i//(sampled_slots.sum())) ,i%(sampled_slots.sum()),:] # parentesis around i//(sampled_slots.sum() are very important



def plot_chains(inferpar, sampler_output, sampled_slots=None, save_to=None):


    fig, axes = plt.subplots(inferpar.get_dim()+1,2, figsize=(15, (inferpar.get_dim()+1)*2))

    labels = list(zip(inferpar.pnames, inferpar.subpops))
    samples = sampler_output.get_chain()
    p_gt = np.load("parameter_ground_truth.npy")
    if sampled_slots is None:
        sampled_slots = [True]*inferpar.get_dim()

    import seaborn as sns
    def plot_chain(frompt,axes):
        ax = axes[0]

        ax.plot(np.arange(frompt,frompt+sampler_output.get_log_prob()[frompt:].shape[0]),
                        sampler_output.get_log_prob()[frompt:,sampled_slots], "navy", alpha=.2, lw=1, label="good walkers")
        ax.plot(np.arange(frompt,frompt+sampler_output.get_log_prob()[frompt:].shape[0]),
                sampler_output.get_log_prob()[frompt:,~sampled_slots], "tomato", alpha=.4, lw=1, label="bad walkers")
        ax.set_title("llik")
        #ax.legend()
        sns.despine(ax=ax, trim=False)
        ax.set_xlim(frompt, frompt+sampler_output.get_log_prob()[frompt:].shape[0])

        #ax.set_xlim(0, len(samples))

        for i in range(inferpar.get_dim()):
            ax = axes[i+1]
            x_plt = np.arange(frompt,frompt+sampler_output.get_log_prob()[frompt:].shape[0])
            ax.plot(x_plt,
                    samples[frompt:,sampled_slots, i], "navy", alpha=.2, lw=1,)
            ax.plot(x_plt,
                    samples[frompt:, ~sampled_slots, i], "tomato", alpha=.4, lw=1,)
            ax.plot(x_plt,
                    np.repeat(p_gt[i],len(x_plt)), "black", alpha=1, lw=2, ls='-.')
            #ax.set_xlim(0, len(samples))
            ax.set_title(labels[i])
            #ax.yaxis.set_label_coords(-0.1, 0.5)
            sns.despine(ax=ax, trim=False)
            ax.set_xlim(frompt, frompt+samples[frompt:].shape[0])
            

        axes[-1].set_xlabel("step number");

    plot_chain(0,axes[:,0])
    plot_chain(3*samples.shape[0]//4,axes[:,1])
    fig.tight_layout()
    if save_to is not None:
        plt.savefig(save_to)

def plot_fit(modinf, loss):
    subpop_names = modinf.subpop_struct.subpop_names
    fig, axes = plt.subplots(len(subpop_names),len(loss.statistics), figsize=(3*len(loss.statistics), 3*len(subpop_names)), sharex=True)
    for j, subpop in enumerate(modinf.subpop_struct.subpop_names):
            gt_s = loss.gt[loss.gt["subpop"]==subpop].sort_index()
            first_date = max(gt_s.index.min(),results[0].index.min())
            last_date = min(gt_s.index.max(), results[0].index.max())
            gt_s = gt_s.loc[first_date:last_date].drop(["subpop"],axis=1).resample("W-SAT").sum()
            
            for i, (stat_name, stat) in enumerate(loss.statistics.items()):
                    ax = axes[j,i]
                    
                    ax.plot(gt_s[stat.data_var], color='k', marker='.', lw=1)
                    for model_df in results:
                            model_df_s = model_df[model_df["subpop"]==subpop].drop(["subpop"],axis=1).loc[first_date:last_date].resample("W-SAT").sum() # todo sub subpop here
                            ax.plot(model_df_s[stat.sim_var],  lw=.9, alpha=.5)
                    #if True:
                    #        init_df_s = outcomes_df_ref[model_df["subpop"]==subpop].drop(["subpop","time"],axis=1).loc[min(gt_s.index):max(gt_s.index)].resample("W-SAT").sum() # todo sub subpop here
                    ax.set_title(f"{stat_name}, {subpop}")
    fig.tight_layout()
    plt.savefig(f"{run_id}_results.pdf")