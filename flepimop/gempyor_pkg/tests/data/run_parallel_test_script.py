"""
Auxiliary runner for pared down version of `gempyor.simulate.simulate` in a new process.

This script is used to test the parallelization of the `run_parallel_SEIR` and
`run_parallel_outcomes` functions by multiprocessing start method. The challenge is that
the start method used by multiprocessing has an impact on the random sampling in these
two functions but setting the start method via `multiprocessing.set_start_method` can
only be done once per a process. To get around this limitation when testing, this script
is used to run a pared down version of `gempyor.simulate.simulate` in a new process with
the desired start method. The script takes the following arguments:

1. `setup_sample_2pop_vaccine_scenarios`: Path to the setup directory.
2. `start_method`: The start method to use for multiprocessing (e.g., 'spawn', 'fork').
3. `n_jobs`: Number of jobs to run in parallel.
4. `do_outcomes`: Whether to run outcomes or not (True/False).

The script sets the working directory to the setup directory, sets the start method for
multiprocessing, and then calls the `main` function with the provided arguments.
The `main` function does the following:

1. Parses the configuration files in the setup directory.
2. Creates a `ModelInfo` object with the parsed configuration.
3. Calls the `run_parallel_SEIR` function with the `ModelInfo` object and the number of
   jobs.
4. If `do_outcomes` is True, calls the `run_parallel_outcomes` function with the
   `ModelInfo` object, the number of jobs, and the simulation index to write.
5. The script is designed to be run as a standalone script, and the `main` function is
   called only if the script is run directly. This allows for easy testing and
   debugging of the script without having to run the entire `gempyor.simulate.simulate`
   function.

This script is part of the `gempyor` package and is used for testing purposes only.
"""

import multiprocessing as mp
import os
from pathlib import Path
import sys

from gempyor.model_info import ModelInfo
from gempyor.outcomes import run_parallel_outcomes
from gempyor.seir import run_parallel_SEIR
from gempyor.shared_cli import parse_config_files


def main(setup_sample_2pop_vaccine_scenarios: Path, n_jobs: int, do_outcomes: bool) -> None:
    """
    Run a pared down version of `gempyor.simulate.simulate` in a new process.

    This function will do as minimal setup as required to call the
    `gempyor.seir.run_parallel_SEIR` function and, optionally, the
    `gempyor.outcomes.run_parallel_outcomes` function.

    Args:
        setup_sample_2pop_vaccine_scenarios: Path to the setup directory.
        n_jobs: Number of jobs to run in parallel.
        do_outcomes: Whether to run outcomes or not.
    """
    cfg = parse_config_files(
        config_filepath=setup_sample_2pop_vaccine_scenarios
        / "config_sample_2pop_vaccine_scenarios.yml",
        id_run_id=None,
        out_run_id=None,
        seir_modifiers_scenarios=[],
        outcome_modifiers_scenarios=[],
        in_prefix=None,
        nslots=None,
        jobs=n_jobs,
        write_csv=False,
        write_parquet=True,
        first_sim_index=1,
        verbose=True,
    )

    seir_modifiers_scenario, outcome_modifiers_scenario = "no_vax", None
    nchains = cfg["nslots"].as_number()
    assert nchains == 10

    modinf = ModelInfo(
        config=cfg,
        nslots=nchains,
        seir_modifiers_scenario=seir_modifiers_scenario,
        outcome_modifiers_scenario=outcome_modifiers_scenario,
        write_csv=cfg["write_csv"].get(bool),
        write_parquet=cfg["write_parquet"].get(bool),
        first_sim_index=cfg["first_sim_index"].get(int),
        in_run_id=cfg["in_run_id"].get(str) if cfg["in_run_id"].exists() else None,
        out_run_id=cfg["out_run_id"].get(str) if cfg["out_run_id"].exists() else None,
        config_filepath=cfg["config_src"].as_str_seq(),
    )

    assert run_parallel_SEIR(modinf, config=cfg, n_jobs=cfg["jobs"].get(int)) is None

    if do_outcomes:
        assert (
            run_parallel_outcomes(
                sim_id2write=cfg["first_sim_index"].get(int),
                modinf=modinf,
                nslots=nchains,
                n_jobs=cfg["jobs"].get(int),
            )
            == 1
        )


if __name__ == "__main__":
    """
    USAGE: run_parallel_test_script.py <setup_sample_2pop_vaccine_scenarios> <start_method> <n_jobs> <do_outcomes>
    <setup_sample_2pop_vaccine_scenarios>: Path to the setup directory.
    <start_method>: The start method to use for multiprocessing (e.g., 'spawn', 'fork').
    <n_jobs>: Number of jobs to run in parallel.
    <do_outcomes>: Whether to run outcomes or not (True/False).
    """
    setup_sample_2pop_vaccine_scenarios = Path(sys.argv[1])
    start_method = sys.argv[2]
    n_jobs = int(sys.argv[3])
    do_outcomes = sys.argv[4].strip().lower() == "true"
    os.chdir(setup_sample_2pop_vaccine_scenarios)
    mp.set_start_method(start_method, force=True)
    main(setup_sample_2pop_vaccine_scenarios, n_jobs, do_outcomes)
