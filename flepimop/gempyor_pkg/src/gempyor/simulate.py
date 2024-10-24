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

import time, warnings, sys

from pathlib import Path
from collections.abc import Iterable

from confuse import Configuration
from click import Context, pass_context

from . import seir, outcomes, model_info, utils
from .shared_cli import config_files_argument, config_file_options, parse_config_files, cli, click_helpstring, mock_context

# from .profile import profile_options

# @profile_options
# @profile()
def simulate(
    config_filepath: Configuration | Path | Iterable[Path],
    id_run_id: str = None,
    out_run_id: str = None,
    seir_modifiers_scenarios: str | Iterable[str] = [],
    outcome_modifiers_scenarios: str | Iterable[str] = [],
    in_prefix: str = None,
    nslots: int = None,
    jobs: int = None,
    write_csv: bool = False,
    write_parquet: bool = True,
    first_sim_index: int = 1,
    stoch_traj_flag: bool = False,
    verbose : bool = True,
) -> int:
    """
    Forward simulate a model using gempyor.

    Args:
        config_filepath: Either a Configuration (in which case: ALL other arguments will be silently ignored) OR
            file path(s) for configuration file(s) (in which case other arguments will be used to override the configuration)
        id_run_id: run_id for the simulation
        out_run_id: run_id for the output
        seir_modifiers_scenarios: scenarios for the SEIR model - if present, used to subset the scenarios in the configuration
        outcome_modifiers_scenarios: scenarios for the outcomes model - if present, used to subset the scenarios in the configuration
        in_prefix: prefix for the input files
        nslots: number of simulation chains
        jobs: amount of parallelization
        write_csv: write output to csv?
        write_parquet: write output to parquet?
        first_sim_index: index of the first simulation
        stoch_traj_flag: stochastic trajectories?
        verbose: print output to console?

    Returns: exit code (side effect: writes output to disk)
    """
    if not isinstance(config_filepath, Configuration):
        largs = locals()
        cfg = parse_config_files(**largs)
    else:
        cfg = config_filepath

    scenarios_combinations = [
        [s, d] for s in (cfg["seir_modifiers"]["scenarios"].as_str_seq() if cfg["seir_modifiers"].exists() else [None])
        for d in (cfg["outcome_modifiers"]["scenarios"].as_str_seq() if cfg["outcome_modifiers"].exists() else [None])]
    
    if verbose:
        print("Combination of modifiers scenarios to be run: ")
        print(scenarios_combinations)
        for seir_modifiers_scenario, outcome_modifiers_scenario in scenarios_combinations:
            print(f"seir_modifier: {seir_modifiers_scenario}, outcomes_modifier: {outcome_modifiers_scenario}")

    nchains = cfg["nslots"].as_number()
    
    if verbose:
        print(f"Simulations to be run: {nchains}")

    for seir_modifiers_scenario, outcome_modifiers_scenario in scenarios_combinations:
        start = time.monotonic()
        if verbose:
            print(f"Running {seir_modifiers_scenario}_{outcome_modifiers_scenario}")

        modinf = model_info.ModelInfo(
            config=cfg,
            nslots=nchains,
            seir_modifiers_scenario=seir_modifiers_scenario,
            outcome_modifiers_scenario=outcome_modifiers_scenario,
            write_csv=cfg["write_csv"].get(bool),
            write_parquet=cfg["write_parquet"].get(bool),
            first_sim_index=cfg["first_sim_index"].get(int),
            in_run_id=cfg["in_run_id"].get(str) if cfg["in_run_id"].exists() else None,
            # in_prefix=config["name"].get() + "/",
            out_run_id=cfg["out_run_id"].get(str) if cfg["out_run_id"].exists() else None,
            # out_prefix=config["name"].get() + "/" + str(seir_modifiers_scenario) + "/" + out_run_id + "/",
            stoch_traj_flag=cfg["stoch_traj_flag"].get(bool),
            config_filepath=cfg["config_src"].as_str_seq(),
        )

        if verbose:
            print(
                f"""
        >> Running from config {cfg["config_src"].as_str_seq()}
        >> Starting {modinf.nslots} model runs beginning from {modinf.first_sim_index} on {cfg["jobs"].get(int)} processes
        >> ModelInfo *** {modinf.setup_name} *** from {modinf.ti} to {modinf.tf}
        >> Running scenario {seir_modifiers_scenario}_{outcome_modifiers_scenario}
        >> running ***{'STOCHASTIC' if cfg["stoch_traj_flag"].get(bool) else 'DETERMINISTIC'}*** trajectories
        """
            )
        # (there should be a run function)
        if cfg["seir"].exists():
            seir.run_parallel_SEIR(modinf, config=cfg, n_jobs=cfg["jobs"].get(int))
        if cfg["outcomes"].exists():
            outcomes.run_parallel_outcomes(sim_id2write=cfg["first_sim_index"].get(int), modinf=modinf, nslots=nchains, n_jobs=cfg["jobs"].get(int))
        if verbose:
            print(
                f">>> {seir_modifiers_scenario}_{outcome_modifiers_scenario} completed in {time.monotonic() - start:.1f} seconds"
            )
        
    return 0

@cli.command(name="simulate", params=[config_files_argument] + list(config_file_options.values()))
@pass_context
def _click_simulate(ctx : Context, **kwargs) -> int:
    """Forward simulate a model using gempyor."""
    cfg = parse_config_files(utils.config, ctx, **kwargs)
    return simulate(cfg)


# will all be removed upon deprecated endpoint removal

import subprocess

def _deprecated_simulate(argv : list[str] = []) -> int:
    if not argv:
        argv = sys.argv[1:]
    clickcmd = ' '.join(['flepimop', 'simulate'] + argv)
    warnings.warn(f"This command is deprecated, use the CLI instead: `{clickcmd}`", DeprecationWarning)
    return subprocess.run(clickcmd, shell=True).returncode

if __name__ == "__main__":
    argv = sys.argv[1:]
    clickcmd = ' '.join(['flepimop', 'simulate'] + argv)
    warnings.warn(f"Use the CLI instead: `{clickcmd}`", DeprecationWarning)
    _deprecated_simulate(argv)

## @endcond
