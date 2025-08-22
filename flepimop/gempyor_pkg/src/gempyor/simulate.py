"""Tools to forward simulate a model with `gempyor`."""

import pickle
import subprocess
import sys
import time
import warnings
from collections.abc import Iterable
from itertools import product
from pathlib import Path
from typing import Any

import click
from confuse import Configuration

from . import outcomes, seir, utils
from .model_info import ModelInfo
from .output import Chains
from .shared_cli import cli, config_file_options, config_files_argument, parse_config_files


def _simulate_seir_and_outcomes(
    modinf: ModelInfo,
    cfg: Configuration,
    chains: Chains | None,
    first_sim_index: int,
    nslots: int,
    jobs: int,
    run_seir: bool = True,
    run_outcomes: bool = True,
) -> None:
    """
    Thin wrapper to run the SEIR and outcomes simulations in parallel.

    Args:
        modinf: A `ModelInfo` instance corresponding to the simulation to be run.
        cfg: A `Configuration` instance containing the simulation configuration.
        chains: Optional `Chains` instance containing MCMC samples to run the simulation
            from. If `None`, the simulation will be run using the parameters in the
            configuration file.
        first_sim_index: The index of the first simulation to be run.
        nslots: The number of simulation chains to be run.
        jobs: The number of parallel jobs to use.
        run_seir: Whether to run the SEIR model.
        run_outcomes: Whether to run the outcomes model.
    """
    if run_seir:
        seir.run_parallel_SEIR(modinf, cfg, n_jobs=jobs)
    if run_outcomes:
        outcomes.run_parallel_outcomes(
            modinf,
            sim_id2write=first_sim_index,
            nslots=nslots,
            n_jobs=jobs,
        )


def simulate(
    config_filepath: Configuration | Path | Iterable[Path],
    from_chains: Path | None = None,
    verbose: bool = True,
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
        verbose: print output to console?

    Returns: exit code (side effect: writes output to disk)
    """
    if not isinstance(config_filepath, Configuration):
        largs = locals()
        largs.pop("verbose")
        largs["config_files"] = largs.pop("config_filepath")
        cfg = parse_config_files(**largs)
    else:
        cfg = config_filepath

    seir_modifiers_scenarios = (
        cfg["seir_modifiers"]["scenarios"].as_str_seq()
        if cfg["seir_modifiers"].exists()
        else [None]
    )
    outcome_modifiers_scenarios = (
        cfg["outcome_modifiers"]["scenarios"].as_str_seq()
        if cfg["outcome_modifiers"].exists()
        else [None]
    )
    scenarios_combinations = list(
        product(seir_modifiers_scenarios, outcome_modifiers_scenarios)
    )

    if verbose:
        print("Combination of modifiers scenarios to be run: ")
        print(scenarios_combinations)
        for seir_modifiers_scenario, outcome_modifiers_scenario in scenarios_combinations:
            print(
                f"seir_modifier: {seir_modifiers_scenario}, "
                f"outcomes_modifier: {outcome_modifiers_scenario}"
            )

    nslots = cfg["nslots"].as_number()

    if verbose:
        print(f"Simulations to be run: {nslots}")

    write_csv = cfg["write_csv"].get(bool)
    write_parquet = cfg["write_parquet"].get(bool)
    first_sim_index = cfg["first_sim_index"].get(int)
    in_run_id = cfg["in_run_id"].get(str) if cfg["in_run_id"].exists() else None
    out_run_id = cfg["out_run_id"].get(str) if cfg["out_run_id"].exists() else None
    config_filepath = cfg["config_src"].as_str_seq()
    n_jobs = cfg["jobs"].get(int)
    run_seir = cfg["seir"].exists()
    run_outcomes = cfg["outcomes"].exists()

    # Load samples from chains if provided
    chains: Chains | None = None
    if from_chains is not None:
        with from_chains.open("rb") as f:
            chains = pickle.load(f)
        if not isinstance(chains, Chains):
            raise ValueError(f"Expected a Chains instance, got {type(chains)}")
        if verbose:
            print(
                f"Loaded chains with shape {chains.shape} "
                f"from {from_chains} for simulating."
            )
        n_chains, n_iterations, _ = chains.shape
        if n_chains != n_jobs:
            raise ValueError(
                f"Number of chains in {from_chains} is {n_chains}, which "
                f"does not match the number of jobs to be run, {n_jobs}."
            )
        if n_iterations < nslots:
            raise ValueError(
                f"Number of iterations in {from_chains} is {n_iterations}, "
                f"which is less than the number of slots to be run, {nslots}."
            )

    for seir_modifiers_scenario, outcome_modifiers_scenario in scenarios_combinations:
        start = time.monotonic()
        if verbose:
            print(f"Running {seir_modifiers_scenario}_{outcome_modifiers_scenario}")

        modinf = ModelInfo(
            config=cfg,
            nslots=nslots,
            seir_modifiers_scenario=seir_modifiers_scenario,
            outcome_modifiers_scenario=outcome_modifiers_scenario,
            write_csv=write_csv,
            write_parquet=write_parquet,
            first_sim_index=first_sim_index,
            in_run_id=in_run_id,
            out_run_id=out_run_id,
            config_filepath=config_filepath,
        )

        if verbose:
            print(f">> Running from config {config_filepath}")
            print(
                f">> Starting {nslots} model runs beginning "
                f"from {first_sim_index} on {n_jobs} processes"
            )
            print(
                f">> ModelInfo *** {modinf.setup_name} "
                f"*** from {modinf.ti} to {modinf.tf}"
            )
            print(
                f">> Running scenario "
                f"{seir_modifiers_scenario}_{outcome_modifiers_scenario}"
            )
            print(f">> using ***{modinf.get_engine()}*** engine for trajectories")

        _simulate_seir_and_outcomes(
            modinf,
            cfg,
            chains,
            first_sim_index,
            nslots,
            n_jobs,
            run_seir=run_seir,
            run_outcomes=run_outcomes,
        )

        if verbose:
            print(
                f">>> {seir_modifiers_scenario}_{outcome_modifiers_scenario} "
                f"completed in {time.monotonic() - start:.1f} seconds"
            )

    return 0


@cli.command(
    name="simulate",
    params=[config_files_argument]
    + list(config_file_options.values())
    + [
        click.Option(
            param_decls=["--from-chains"],
            type=click.Path(exists=True, dir_okay=False),
            default=None,
            show_default=True,
            help=(
                "Optional path to a chains pickle file to run simulations from. "
                "Will override the modifiers within the config file(s)."
            ),
        )
    ],
    context_settings=dict(help_option_names=["-h", "--help"]),
)
@click.pass_context
def _click_simulate(ctx: click.Context, **kwargs: Any) -> int:
    """Forward simulate a model using gempyor."""
    cfg = parse_config_files(utils.config, ctx, **kwargs)
    return simulate(cfg, from_chains=kwargs.get("from_chains"))


def _deprecated_simulate(argv: list[str] | None = None) -> int:
    argv = argv or []
    if not argv:
        argv = sys.argv[1:]
    clickcmd = " ".join(["flepimop", "simulate"] + argv)
    warnings.warn(
        f"This command is deprecated, use the CLI instead: `{clickcmd}`", DeprecationWarning
    )
    return subprocess.run(clickcmd, shell=True).returncode


if __name__ == "__main__":
    argv = sys.argv[1:]
    clickcmd = " ".join(["flepimop", "simulate"] + argv)
    warnings.warn(f"Use the CLI instead: `{clickcmd}`", DeprecationWarning)
    _deprecated_simulate(argv)
