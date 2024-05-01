#!/usr/bin/env python
import click
from gempyor import model_info, file_paths, config, inference_parameter
import gempyor.inference
from gempyor.utils import config, as_list
import gempyor
import numpy as np
import os, shutil, copy
import emcee
import multiprocessing

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
    config.clear()
    config.read(user=False)
    config.set_file(project_path + config_filepath)

    # Compute the list of scenarios to run. Since multiple = True, it's always a list.
    if not seir_modifiers_scenarios:
        seir_modifiers_scenarios = None
        if config["seir_modifiers"].exists():
            if config["seir_modifiers"]["scenarios"].exists():
                seir_modifiers_scenarios = config["seir_modifiers"]["scenarios"].as_str_seq()
        # Model Info handles the case of the default scneario
    if not outcome_modifiers_scenarios:
        outcome_modifiers_scenarios = None
        if config["outcomes"].exists() and config["outcome_modifiers"].exists():
            if config["outcome_modifiers"]["scenarios"].exists():
                outcome_modifiers_scenarios = config["outcome_modifiers"]["scenarios"].as_str_seq()

    outcome_modifiers_scenarios = as_list(outcome_modifiers_scenarios)
    seir_modifiers_scenarios = as_list(seir_modifiers_scenarios)
    if len(seir_modifiers_scenarios) != 1 or len(outcome_modifiers_scenarios) != 1:
        raise ValueError(
            f"Only support configurations files with one scenario, got"
            f"seir: {seir_modifiers_scenarios}"
            f"outcomes: {outcome_modifiers_scenarios}"
        )

    scenarios_combinations = [[s, d] for s in seir_modifiers_scenarios for d in outcome_modifiers_scenarios]
    for seir_modifiers_scenario, outcome_modifiers_scenario in scenarios_combinations:
        print(f"seir_modifier: {seir_modifiers_scenario}, outcomes_modifier:{outcome_modifiers_scenario}")

    if not nwalkers:
        nwalkers = config["nslots"].as_number()  # TODO
    print(f"Number of walkers be run: {nwalkers}")

    for seir_modifiers_scenario, outcome_modifiers_scenario in scenarios_combinations:
        print(f"Running {seir_modifiers_scenario}_{outcome_modifiers_scenario}")
        if prefix is None:
            prefix = config["name"].get() + "/" + run_id + "/"

        write_csv = False
        write_parquet = True

        modinf = model_info.ModelInfo(
            config=config,
            nslots=1,
            seir_modifiers_scenario=seir_modifiers_scenario,
            outcome_modifiers_scenario=outcome_modifiers_scenario,
            write_csv=write_csv,
            write_parquet=write_parquet,
            first_sim_index=0,
            in_run_id=run_id,
            in_prefix=prefix,
            out_run_id=run_id,
            out_prefix=prefix,
            inference_filename_prefix="emcee",
            inference_filepath_suffix="",
            stoch_traj_flag=False,
            config_filepath=config_filepath,
        )

        print(
            f"""
    >> Running from config {config_filepath}
    >> Starting {modinf.nslots} model runs beginning from {modinf.first_sim_index} on {jobs} processes
    >> ModelInfo *** {modinf.setup_name} *** from {modinf.ti} to {modinf.tf}
    >> Running scenario {seir_modifiers_scenario}_{outcome_modifiers_scenario}
    """
        )

    inferpar = inference_parameter.InferenceParameters(global_config=config, modinf=modinf)
    p0 = inferpar.draw_initial(n_draw=nwalkers)
    for i in range(nwalkers):
        assert inferpar.check_in_bound(
            proposal=p0[i]
        ), "The initial parameter draw is not within the bounds, check the perturbation distributions"

    loss = logloss.LogLoss(inference_config=config["inference"], data_dir=project_path, modinf=modinf)

    static_sim_arguments = gempyor.inference.get_static_arguments(modinf=modinf)

    test_run = True
    if test_run:
        ss = copy.deepcopy(static_sim_arguments)
        ss["snpi_df_in"] = ss["snpi_df_ref"]
        ss["hnpi_df_in"] = ss["hnpi_df_ref"]
        # delete the ref
        del ss["snpi_df_ref"]
        del ss["hnpi_df_ref"]

        hosp = gempyor.inference.simulation_atomic(**ss, modinf=modinf)

        ll_total, logloss, regularizations = loss.compute_logloss(model_df=hosp, modinf=modinf)
        print(
            f"test run successful 🎉, with logloss={ll_total:.1f} including {regularizations:.1f} for regularization ({regularizations/ll_total*100:.1f}%) "
        )

    filename = f"{run_id}_backend.h5"
    backend = emcee.backends.HDFBackend(filename)
    # TODO here for resume

    if resume:
        p0 = None
    else:
        backend.reset(nwalkers, inferpar.get_dim())
        p0 = p0

    moves = [(emcee.moves.StretchMove(live_dangerously=True), 1)]
    with multiprocessing.Pool(ncpu) as pool:
        sampler = emcee.EnsembleSampler(
            nwalkers,
            inferpar.get_dim(),
            gempyor.inference.emcee_logprob,
            args=[modinf, inferpar, loss, static_sim_arguments],
            pool=pool,
            backend=backend,
            moves=moves,
        )
        state = sampler.run_mcmc(p0, niter, progress=True, skip_initial_state_check=True)

    print(f"Done, mean acceptance fraction: {np.mean(sampler.acceptance_fraction):.3f}")

    sampled_slots = gempyor.inference.find_walkers_to_sample()

    # plotting the chain
    sampler = emcee.backends.HDFBackend(filename, read_only=True)
    gempyor.inference.plot_chains(
        inferpar=inferpar, sampler_output=sampler, sampled_slots=sampled_slots, save_to=f"{run_id}_chains.pdf"
    )
    print("EMCEE Run done, doing sampling")

    position_arguments = [modinf, inferpar, loss, static_sim_arguments, True]
    shutil.rmtree("model_output/")
    with Pool(ncpu) as pool:
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
