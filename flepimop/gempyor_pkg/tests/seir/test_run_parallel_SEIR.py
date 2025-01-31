import multiprocessing as mp
import os
from pathlib import Path
import shutil
import subprocess

import pandas as pd
import pytest

from gempyor.testing import run_test_in_separate_process


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
    assert (
        run_test_in_separate_process(
            Path(__file__).parent / "run_parallel_SEIR_test_script.py",
            setup_sample_2pop_vaccine_scenarios / "test.py",
            args=[str(setup_sample_2pop_vaccine_scenarios), start_method, str(n_jobs)],
        )
        is None
    )

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
