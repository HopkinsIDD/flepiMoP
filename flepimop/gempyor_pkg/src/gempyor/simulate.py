#!/usr/bin/env python

##
# @file
# @brief Runs hospitalization model
#
# @details
#
# ## Configuration Items
#
# ```yaml
# name: <string>
# setup_name: <string>
# start_date: <date>
# end_date: <date>
# dt: float
# nslots: <integer> overridden by the -n/--nslots script parameter
# data_path: <path to directory>
# subpop_setup:
#   geodata: <path to file>
#   mobility: <path to file>
#
# seir:
#   parameters
#     alpha: <float>
#     sigma: <float>
#     gamma: <random distribution>
#     R0s: <random distribution>
#
# seir_modifiers:
#   scenarios:
#     - <scenario 1 name>
#     - <scenario 2 name>
#     - ...
#   settings:
#     <scenario 1 name>:
#       method: choose one - "SinglePeriodModifier", ", "StackedModifier"
#       ...
#     <scenario 2 name>:
#       method: choose one - "SinglePeriodModifier", "", "StackedModifier"
#       ...
#
# seeding:
#   method: choose one - "PoissonDistributed", "FolderDraw"
# ```
#
# ### seir_modifiers::scenarios::settings::<scenario name>
#
# If {method} is
# ```yaml
# seir_modifiers:
#   scenarios:
#     <scenario name>:
#       method: SinglePeriodModifier
#       parameter: choose one - "alpha, sigma, gamma, r0"
#       period_start_date: <date>
#       period_end_date: <date>
#       value: <random distribution>
#       subpop: <list of strings> optional
# ```
#
# If {method} is
# ```yaml
# seir_modifiers:
#   scenarios:
#     <scenario name>:
#       method:
#       period_start_date: <date>
#       period_end_date: <date>
#       value: <random distribution>
#       subpop: <list of strings> optional
# ```
#
# If {method} is StackedModifier
# ```yaml
# seir_modifiers:
#   scenarios:
#     <scenario name>:
#       method: StackedModifier
#       scenarios: <list of scenario names>
# ```
#
# ### seeding
#
# If {seeding::method} is PoissonDistributed
# ```yaml
# seeding:
#   method: PoissonDistributed
#   lambda_file: <path to file>
# ```
#
# If {seeding::method} is FolderDraw
# ```yaml
# seeding:
#   method: FolderDraw
#   folder_path: \<path to dir\>; make sure this ends in a '/'
# ```
#
# ## Input Data
#
# * <b>{data_path}/{subpop_setup::geodata}</b> is a csv with columns {subpop_setup::subpop_names} and {subpop_setup::subpop_pop}
# * <b>{data_path}/{subpop_setup::mobility}</b>
#
# If {seeding::method} is PoissonDistributed
# * {seeding::lambda_file}
#
# If {seeding::method} is FolderDraw
# * {seeding::folder_path}/[simulation ID].impa.csv
#
# ## Output Data
#
# * model_output/{setup_name}_[scenario]/[simulation ID].seir.[csv/parquet]
# * model_parameters/{setup_name}_[scenario]/[simulation ID].spar.[csv/parquet]
# * model_parameters/{setup_name}_[scenario]/[simulation ID].snpi.[csv/parquet]
# ## Configuration Items
#
# ```yaml
# outcomes:
#  method: delayframe                   # Only fast is supported atm. Makes fast delay_table computations. Later agent-based method ?
#  paths:
#    param_from_file: TRUE               #
#    param_subpop_file: <path.csv>       # OPTIONAL: File with param per csv. For each param in this file
#  scenarios:                           # Outcomes scenarios to run
#    - low_death_rate
#    - mid_death_rate
#  settings:                            # Setting for each scenario
#    low_death_rate:
#      new_comp1:                               # New compartement name
#        source: incidence                      # Source of the new compartement: either an previously defined compartement or "incidence" for diffI of the SEIR
#        probability:  <random distribution>           # Branching probability from source
#        delay: <random distribution>                  # Delay from incidence of source to incidence of new_compartement
#        duration: <random distribution>               # OPTIONAL ! Duration in new_comp. If provided, the model add to it's
#                                                      #output "new_comp1_curr" with current amount in new_comp1
#      new_comp2:                               # Example for a second compatiment
#        source: new_comp1
#        probability: <random distribution>
#        delay: <random distribution>
#        duration: <random distribution>
#      death_tot:                               # Possibility to combine compartements for death.
#        sum: ['death_hosp', 'death_ICU', 'death_incid']
#
#    mid_death_rate:
#      ...
#
# ## Input Data
#
# * <b>{param_subpop_file}</b> is a csv with columns subpop, parameter, value. Parameter is constructed as, e.g for comp1:
#                probability: Pnew_comp1|source
#                delay:       Dnew_comp1
#                duration:    Lnew_comp1


# ## Output Data
# * {output_path}/model_output/{setup_name}_[scenario]/[simulation ID].hosp.parquet


## @cond

import multiprocessing
import time, os, itertools

import click

from gempyor import seir, outcomes, model_info, file_paths
from gempyor.utils import config, as_list

# from .profile import profile_options


@click.command()
@click.option(
    "-c",
    "--config",
    "config_file",
    envvar=["CONFIG_PATH", "CONFIG_PATH"],
    type=click.Path(exists=True),
    required=True,
    help="configuration file for this simulation",
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
    envvar="FLEPI_NUM_SLOTS",
    type=click.IntRange(min=1),
    help="override the # of simulation runs in the config file",
)
@click.option(
    "-i",
    "--first_sim_index",
    envvar="FIRST_SIM_INDEX",
    type=click.IntRange(min=1),
    default=1,
    show_default=True,
    help="The index of the first simulation",
)
@click.option(
    "-j",
    "--jobs",
    envvar="FLEPI_NJOBS",
    type=click.IntRange(min=1),
    default=multiprocessing.cpu_count(),
    show_default=True,
    help="the parallelization factor",
)
@click.option(
    "--stochastic/--non-stochastic",
    "--stochastic/--non-stochastic",
    "stoch_traj_flag",
    envvar="FLEPI_STOCHASTIC_RUN",
    type=bool,
    default=False,
    help="Flag determining whether to run stochastic simulations or not",
)
@click.option(
    "--in-id",
    "--in-id",
    "in_run_id",
    envvar="FLEPI_RUN_INDEX",
    type=str,
    default=file_paths.run_id(),
    show_default=True,
    help="Unique identifier for the run",
)  # Default does not make sense here
@click.option(
    "--out-id",
    "--out-id",
    "out_run_id",
    envvar="FLEPI_RUN_INDEX",
    type=str,
    default=file_paths.run_id(),
    show_default=True,
    help="Unique identifier for the run",
)
@click.option(
    "--in-prefix",
    "--in-prefix",
    "in_prefix",
    envvar="FLEPI_PREFIX",
    type=str,
    default=None,
    show_default=True,
    help="unique identifier for the run",
)
@click.option(
    "--write-csv/--no-write-csv",
    default=False,
    show_default=True,
    help="write CSV output at end of simulation",
)
@click.option(
    "--write-parquet/--no-write-parquet",
    default=True,
    show_default=True,
    help="write parquet file output at end of simulation",
)
# @profile_options
def simulate(
    config_file,
    in_run_id,
    out_run_id,
    seir_modifiers_scenarios,
    outcome_modifiers_scenarios,
    in_prefix,
    nslots,
    jobs,
    write_csv,
    write_parquet,
    first_sim_index,
    stoch_traj_flag,
):
    config.clear()
    config.read(user=False)
    config.set_file(config_file)
    print(outcome_modifiers_scenarios, seir_modifiers_scenarios)

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
                outcome_modifiers_scenarios = config["outcomes"]["scenarios"].as_str_seq()

    outcome_modifiers_scenarios = as_list(outcome_modifiers_scenarios)
    seir_modifiers_scenarios = as_list(seir_modifiers_scenarios)
    print(outcome_modifiers_scenarios, seir_modifiers_scenarios)

    scenarios_combinations = [[s, d] for s in seir_modifiers_scenarios for d in outcome_modifiers_scenarios]
    print("Combination of modifiers scenarios to be run: ")
    print(scenarios_combinations)
    for seir_modifiers_scenario, outcome_modifiers_scenario in scenarios_combinations:
        print(f"seir_modifier: {seir_modifiers_scenario}, outcomes_modifier:{outcome_modifiers_scenario}")

    if not nslots:
        nslots = config["nslots"].as_number()
    print(f"Simulations to be run: {nslots}")

    for seir_modifiers_scenario, outcome_modifiers_scenario in scenarios_combinations:
        start = time.monotonic()
        print(f"Running {seir_modifiers_scenario}_{outcome_modifiers_scenario}")

        modinf = model_info.ModelInfo(
            config=config,
            nslots=nslots,
            seir_modifiers_scenario=seir_modifiers_scenario,
            outcome_modifiers_scenario=outcome_modifiers_scenario,
            write_csv=write_csv,
            write_parquet=write_parquet,
            first_sim_index=first_sim_index,
            in_run_id=in_run_id,
            # in_prefix=config["name"].get() + "/",
            out_run_id=out_run_id,
            # out_prefix=config["name"].get() + "/" + str(seir_modifiers_scenario) + "/" + out_run_id + "/",
            stoch_traj_flag=stoch_traj_flag,
        )

        print(
            f"""
    >> Running from config {config_file}
    >> Starting {modinf.nslots} model runs beginning from {modinf.first_sim_index} on {jobs} processes
    >> ModelInfo *** {modinf.setup_name} *** from {modinf.ti} to {modinf.tf}
    >> Running scenario {seir_modifiers_scenario}_{outcome_modifiers_scenario}
    >> running ***{'STOCHASTIC' if stoch_traj_flag else 'DETERMINISTIC'}*** trajectories
    """
        )
        # (there should be a run function)
        if config["seir"].exists():
            seir.run_parallel_SEIR(modinf, config=config, n_jobs=jobs)
        if config["outcomes"].exists():
            outcomes.run_parallel_outcomes(sim_id2write=first_sim_index, modinf=modinf, nslots=nslots, n_jobs=jobs)
        print(
            f">>> {seir_modifiers_scenario}_{outcome_modifiers_scenario} completed in {time.monotonic() - start:.1f} seconds"
        )


if __name__ == "__main__":
    simulate()

## @endcond
