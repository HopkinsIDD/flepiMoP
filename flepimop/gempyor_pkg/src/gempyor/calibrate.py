#!/usr/bin/env python
import click
from gempyor import model_info, file_paths, config, inference_parameter
from gempyor.inference import GempyorInference
from gempyor.utils import config, as_list
import gempyor
import numpy as np
import os, shutil, copy
import emcee
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
    type=click.Path(exists=True),
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
    "-s",
    "--seir_modifiers_scenario",
    "seir_modifiers_scenarios",
    envvar="FLEPI_SEIR_SCENARIO",
    type=str,
    default=[],
    multiple=True,
    help="override the NPI scenario(s) run for this simulation [supports multiple NPI scenarios: `-s Wuhan -s None`]",
)
@click.option(
    "-d",
    "--outcome_modifiers_scenario",
    "outcome_modifiers_scenarios",
    envvar="FLEPI_OUTCOME_SCENARIO",
    type=str,
    default=[],
    multiple=True,
    help="Scenario of outcomes to run",
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
    "ninter",
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
    "run_id",
    envvar="FLEPI_RUN_INDEX",
    type=str,
    default=file_paths.run_id(),
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
    seir_modifiers_scenarios,
    outcome_modifiers_scenarios,
    nwalkers,
    niter,
    nsamples,
    nthin,
    ncpu,
    run_id,
    prefix,
    resume,
    resume_location,
):
    gempyor_inference = GempyorInference(
            config_filepath=config_filepath,
            run_id=run_id,
            prefix=None,
            first_sim_index=1,
            stoch_traj_flag=False,
            rng_seed=None,
            nslots=1,
            inference_filename_prefix="",  # usually for {global or chimeric}/{intermediate or final}
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

    filename = f"{run_id}_backend.h5"
    if os.path.exists(filename):
        if not resume:
            print(f"File {filename} already exists, remove it or use --resume")
            return
    else:
        print(f"writing to {filename}")
    
    # TODO here for resume
    if resume or resume_location is not None:
        print("Doing a resume, this only work with the same number of slot and parameters right now")
        p0 = None
        if resume_location is not None:
            backend = emcee.backends.HDFBackend(resume_location)
        else:
            if not os.path.exists(filename):
                print(f"File {filename} does not exist, cannot resume")
                return
            backend = emcee.backends.HDFBackend(filename)

    else:
        backend = emcee.backends.HDFBackend(filename)
        backend.reset(nwalkers, gempyor_inference.inferpar.get_dim())
        p0 = gempyor_inference.inferpar.draw_initial(n_draw=nwalkers)
        for i in range(nwalkers):
            assert gempyor_inference.inferpar.check_in_bound(
                proposal=p0[i]
            ), "The initial parameter draw is not within the bounds, check the perturbation distributions"

    moves = [(emcee.moves.StretchMove(live_dangerously=True), 1)]
    with multiprocessing.Pool(ncpu) as pool:
        sampler = emcee.EnsembleSampler(
            nwalkers,
            gempyor_inference.inferpar.get_dim(),
            gempyor.inference.emcee_logprob,
            pool=pool,
            backend=backend,
            moves=moves,
        )
        state = sampler.run_mcmc(p0, niter, progress=True, skip_initial_state_check=True)

    print(f"Done, mean acceptance fraction: {np.mean(sampler.acceptance_fraction):.3f}")

    # plotting the chain
    sampler = emcee.backends.HDFBackend(filename, read_only=True)
    gempyor.inference.plot_chains(
        inferpar=gempyor_inference.inferpar, sampler_output=sampler, sampled_slots=np.ones(), save_to=f"{run_id}_chains.pdf"
    )
    print("EMCEE Run done, doing sampling")

    shutil.rmtree("model_output/")
    with multiprocessing.Pool(ncpu) as pool:
        results = pool.starmap(
            gempyor.inference.emcee_logprob, [(sample, *position_arguments) for sample in exported_samples]
        )
    results = []
    for fn in gempyor.utils.list_filenames(folder="model_output/", filters=[run_id, "hosp.parquet"]):
        df = gempyor.read_df(fn)
        df = df.set_index("date")
        results.append(df)


if __name__ == "__main__":
    calibrate()

## @endcond
