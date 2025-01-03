import multiprocessing as mp
import os
from pathlib import Path
import shutil
import subprocess

import pandas as pd
import pytest


@pytest.fixture
def setup_sample_2pop_vaccine_scenarios(tmp_path: Path) -> Path:
    tutorials_path = Path(os.path.dirname(__file__) + "/../../../../examples/tutorials")
    for file in (
        "config_sample_2pop_vaccine_scenarios.yml",
        "model_input/geodata_sample_2pop.csv",
        "model_input/mobility_sample_2pop.csv",
        "model_input/ic_2pop.csv",
    ):
        source = tutorials_path / file
        destination = tmp_path / file
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(source, destination)
    return tmp_path


@pytest.mark.parametrize("n_jobs", (1, 2))
@pytest.mark.parametrize("start_method", mp.get_all_start_methods())
def test_run_parallel_SEIR_by_multiprocessing_start_method(
    monkeypatch: pytest.MonkeyPatch,
    setup_sample_2pop_vaccine_scenarios: Path,
    n_jobs: int,
    start_method: str,
) -> None:
    """
    Test the parallelization of `run_parallel_SEIR` by multiprocessing start method.

    This test:

    1. Sets up the test environment by copying the necessary files to a temporary directory.
    2. Runs a pared down version of `gempyor.simulate.simulate` in a new process.
    3. Reads the contents of the 'spar' directory as a DataFrame.
    4. Tests the contents of the 'spar' DataFrame.

    The reason for the new process is to control the start method used by multiprocessing.
    The `run_parallel_SEIR` function behaves differently depending on the start method used.
    Under the hood `tqdm.contrib.concurrent.process_map` creates a
    `concurrent.futures.ProcessPoolExecutor` with the default start method (see
    [tqdm/tqdm#1265](https://github.com/tqdm/tqdm/pull/1265)), which is 'spawn' on
    MacOS/Windows and 'fork' on Linux. The work around to this is to force multiprocessing
    to use the desired start method by setting it in the '__main__' module with
    [`multiprocessing.set_start_method`](https://docs.python.org/3.11/library/multiprocessing.html#multiprocessing.set_start_method).
    """
    # Test setup
    monkeypatch.chdir(setup_sample_2pop_vaccine_scenarios)

    # Run a pared down version of `gempyor.simulate.simulate` in a new process
    test_python_script = setup_sample_2pop_vaccine_scenarios / "test.py"
    with open(test_python_script, "w") as f:
        f.write(
            f"""
import multiprocessing as mp
import os
from pathlib import Path

from gempyor.model_info import ModelInfo
from gempyor.seir import run_parallel_SEIR
from gempyor.shared_cli import parse_config_files

def main():
    setup_sample_2pop_vaccine_scenarios = Path("{setup_sample_2pop_vaccine_scenarios}")

    cfg = parse_config_files(
        config_filepath=setup_sample_2pop_vaccine_scenarios
        / "config_sample_2pop_vaccine_scenarios.yml",
        id_run_id=None,
        out_run_id=None,
        seir_modifiers_scenarios=[],
        outcome_modifiers_scenarios=[],
        in_prefix=None,
        nslots=None,
        jobs={n_jobs},
        write_csv=False,
        write_parquet=True,
        first_sim_index=1,
        stoch_traj_flag=False,
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
        stoch_traj_flag=cfg["stoch_traj_flag"].get(bool),
        config_filepath=cfg["config_src"].as_str_seq(),
    )

    assert run_parallel_SEIR(modinf, config=cfg, n_jobs=cfg["jobs"].get(int)) is None

if __name__ == "__main__":
    os.chdir("{setup_sample_2pop_vaccine_scenarios}")
    mp.set_start_method("{start_method}", force=True)
    main()
"""
        )

    python = shutil.which("python")
    assert python is not None
    proc = subprocess.run([python, test_python_script], capture_output=True, check=True)
    assert (
        proc.returncode == 0
    ), f"Issue running test script returned {proc.returncode}: {proc.stderr.decode()}."

    # Get the contents of 'spar' directories as DataFrames
    spar_directory: Path | None = None
    for p in setup_sample_2pop_vaccine_scenarios.rglob("*"):
        if p.is_dir() and p.name == "spar":
            spar_directory = p
        if spar_directory is not None:
            break

    def read_directory(directory: Path) -> list[pd.DataFrame]:
        dfs: list[pd.DataFrame] | pd.DataFrame = []
        for i, f in enumerate(sorted(list(directory.glob("*.parquet")))):
            dfs.append(pd.read_parquet(f))
            dfs[-1]["slot"] = i
        dfs = pd.concat(dfs)
        return dfs

    spar = read_directory(spar_directory)

    # Test contents of 'spar' DataFrames
    assert spar[spar["parameter"] == "Ro"]["value"].nunique() == 10
