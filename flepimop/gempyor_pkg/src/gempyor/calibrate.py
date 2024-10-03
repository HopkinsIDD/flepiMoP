#!/usr/bin/env python
import click
from gempyor import model_info, file_paths, config, inference_parameter
from gempyor.inference import GempyorInference
from gempyor.utils import config, as_list
import gempyor
import numpy as np
import os, shutil, copy
import emcee
import pathlib
import multiprocessing
import gempyor.postprocess_inference

# from .profile import profile_options

# disable  operations using the MKL linear algebra.
os.environ["OMP_NUM_THREADS"] = "1"


@click.command()
@click.option(
    "-c",
    "--config",
    "config_filepath",
    envvar="CONFIG_PATH",
    type=click.Path(),
    required=True,
    help="configuration file for this simulation",
)
@click.option(
    "-p",
    "--project_path",
    "project_path",
    envvar="PROJECT_PATH",
    type=click.Path(exists=True),
    default=".",
    required=True,
    help="path to the flepiMoP directory",
)
@click.option(
    "-n",
    "--nslots",
    "--nwalkers",
    "nwalkers",
    envvar="FLEPI_NUM_SLOTS",
    type=click.IntRange(min=1),
    help="override the # of walkers simulation runs in the config file",
)
@click.option(
    "--niterations",
    "niter",
    type=click.IntRange(min=1),
    help="override the # of samples to produce simulation runs in the config file",
)
@click.option(
    "--nsamples",
    "nsamples",
    type=click.IntRange(min=1),
    help="override the # of samples to produce simulation runs in the config file",
)
@click.option(
    "--nthin",
    "nthin",
    type=click.IntRange(min=5),
    help="override the # of samples to thin",
)
@click.option(
    "-j",
    "--jobs",
    "ncpu",
    envvar="FLEPI_NJOBS",
    type=click.IntRange(min=1),
    default=multiprocessing.cpu_count(),
    show_default=True,
    help="the parallelization factor",
)
@click.option(
    "--id",
    "--id",
    "input_run_id",
    envvar="FLEPI_RUN_INDEX",
    type=str,
    default=None,
    help="Unique identifier for this run",
)
@click.option(
    "-prefix",
    "--prefix",
    "prefix",
    envvar="FLEPI_PREFIX",
    type=str,
    default=None,
    show_default=True,
    help="unique identifier for the run",
)
@click.option(
    "--resume/--no-resume",
    "resume",
    envvar="FLEPI_RESUME",
    type=bool,
    default=False,
    help="Flag determining whether to resume or not the current calibration.",
)
@click.option(
    "-r",
    "--resume_location",
    "--resume_location",
    type=str,
    default=None,
    envvar="RESUME_LOCATION",
    help="The location (folder or an S3 bucket) to use as the initial to the first block of the current run",
)
# @profile_options
# @profile()
def calibrate(
    config_filepath,
    project_path,
    nwalkers,
    niter,
    nsamples,
    nthin,
    ncpu,
    input_run_id,
    prefix,
    resume,
    resume_location,
):
    # Choose a run_id
    if input_run_id is None:
        base_run_id = pathlib.Path(config_filepath).stem.replace("config_", "")
        run_id = f"{base_run_id}-{file_paths.run_id()}"
        print(f"Auto-generating run_id: {run_id}")
    else:
        run_id = input_run_id

    # Select a file name and create the backend/resume, and create the initial parameters
    if resume or resume_location is not None:
        if resume_location is None:
            filename = f"{run_id}_backend.h5"
        else:
            filename = resume_location

        if not os.path.exists(filename):
            print(f"File {filename} does not exist, cannot resume")
            return
        print(f"Doing a resume from {filename}, this only work with the same number of slot/walkers and parameters right now")
        backend = emcee.backends.HDFBackend(filename)

        # Normally one would put p0 = None to get the last State from the sampler, but that poses problems when the likelihood change
        # and then acceptances are not guaranted, see issue #316. This solves this issue and greates a new chain with llik evaluation
        p0 = backend.get_last_sample().coords
    else:
        filename = f"{run_id}_backend.h5"
        if os.path.exists(filename):
            if not resume:
                print(f"File {filename} already exists, remove it or use --resume")
                return
        backend = emcee.backends.HDFBackend(filename)
        backend.reset(nwalkers, gempyor_inference.inferpar.get_dim())
        p0 = gempyor_inference.inferpar.draw_initial(n_draw=nwalkers)
        for i in range(nwalkers):
            assert gempyor_inference.inferpar.check_in_bound(
                proposal=p0[i]
            ), "The initial parameter draw is not within the bounds, check the perturbation distributions"
    
    gempyor_inference = GempyorInference(
        config_filepath=config_filepath,
        run_id=run_id,
        prefix=None,
        first_sim_index=1,
        stoch_traj_flag=False,
        rng_seed=None,
        nslots=1,
        inference_filename_prefix="global/final/",  # usually for {global or chimeric}/{intermediate or final}
        inference_filepath_suffix="",  # usually for the slot_id
        out_run_id=None,  # if out_run_id is different from in_run_id, fill this
        out_prefix=None,  # if out_prefix is different from in_prefix, fill this
        path_prefix=project_path,  # in case the data folder is on another directory
        autowrite_seir=False,
    )

    if not nwalkers:
        nwalkers = config["nslots"].as_number()  # TODO
    print(f"Number of walkers be run: {nwalkers}")

    test_run = True

    if test_run:
        gempyor_inference.perform_test_run()
        # Make a plot of the runs directly from config
        print(f"Making {nwalkers//2} simulations from config to plot")
        with multiprocessing.Pool(ncpu) as pool:
            results = pool.starmap(
                gempyor_inference.simulate_proposal, [p0[i] for i in range(nwalkers//2)]
            )
        gempyor.postprocess_inference.plot_fit(modinf=gempyor_inference.modinf, 
                                            loss=gempyor_inference.logloss,
                                            plot_projections=True,
                                            list_of_df=results, save_to=f"{run_id}_config.pdf")


    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # @JOSEPH: find below a "cocktail" move proposal 
    moves = [(emcee.moves.DEMove(), 0.5*0.9*0.9),
             (emcee.moves.DEMove(gamma0=1.0),0.5*0.9*0.1),
             (emcee.moves.DESnookerMove(),0.5*0.1),                     # First three moves: DEMove --> DE is good at "optimizing". Moves based on the (really great!) discussion in https://groups.google.com/g/emcee-users/c/FCAq459Y9OE
             (emcee.moves.StretchMove(live_dangerously=True), 0.25),    # Stretch gives good chain movement
             (emcee.moves.KDEMove(bw_method='scott'), 0.25)]            # Based on personal experience with pySODM (Tijs) - KDEMove works really well but I think it's important for this one to have at least 3x more walkers than parameters.
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    moves = [(emcee.moves.StretchMove(live_dangerously=True), 1)]
    gempyor_inference.set_silent(False)
    
    #with multiprocessing.Pool(ncpu) as pool:
    #    sampler = emcee.EnsembleSampler(
    #        nwalkers,
    #        gempyor_inference.inferpar.get_dim(),
    #        gempyor_inference.get_logloss_as_single_number,
    #        pool=pool,
    #        backend=backend,
    #        moves=moves,
    #    )
    #    state = sampler.run_mcmc(p0, niter, progress=True, skip_initial_state_check=True)

    # hack around memory management: run by batch of 10 iterations
    nbatch = 10
    for i in range(niter // nbatch):
        if i == 0:
            start_val = p0
        else:
            start_val = None

        with multiprocessing.Pool(ncpu) as pool:
            sampler = emcee.EnsembleSampler(
                nwalkers,
                gempyor_inference.inferpar.get_dim(),
                gempyor_inference.get_logloss_as_single_number,
                pool=pool,
                backend=backend,
                moves=moves,
            )
            state = sampler.run_mcmc(start_val, nbatch, progress=True, skip_initial_state_check=True)
    print(f"Done, mean acceptance fraction: {np.mean(sampler.acceptance_fraction):.3f}")

    # plotting the chain
    sampler = emcee.backends.HDFBackend(filename, read_only=True)
    gempyor.postprocess_inference.plot_chains(
        inferpar=gempyor_inference.inferpar, chains = sampler.get_chain(), llik = sampler.get_log_prob(), sampled_slots=None, save_to=f"{run_id}_chains.pdf"
    )
    print("EMCEE Run done, doing sampling")

    shutil.rmtree("model_output/", ignore_errors=True)
    shutil.rmtree(os.path.join(project_path, "model_output/"), ignore_errors=True)

    max_indices = np.argsort(sampler.get_log_prob()[-1, :])[-nsamples:]
    samples = sampler.get_chain()[-1, max_indices, :]  # the last iteration, for selected slots
    gempyor_inference.set_save(True)
    with multiprocessing.Pool(ncpu) as pool:
        results = pool.starmap(
            gempyor_inference.get_logloss_as_single_number, [(samples[i, :],) for i in range(len(max_indices))]
        )

    results = []
    for fn in gempyor.utils.list_filenames(folder=os.path.join(project_path,"model_output/"), filters=[run_id, "hosp.parquet"]):
        df = gempyor.read_df(fn)
        df = df.set_index("date")
        results.append(df)

    gempyor.postprocess_inference.plot_fit(modinf=gempyor_inference.modinf, 
                                           loss=gempyor_inference.logloss, 
                                           list_of_df=results, save_to=f"{run_id}_fit.pdf")
    
    gempyor.postprocess_inference.plot_fit(modinf=gempyor_inference.modinf, 
                                           loss=gempyor_inference.logloss,
                                           plot_projections=True,
                                           list_of_df=results, save_to=f"{run_id}_fit_w_proj.pdf")


if __name__ == "__main__":
    calibrate()

## @endcond
