import multiprocessing as mp
import os
from pathlib import Path

import pytest
import yaml

from gempyor.testing import run_test_in_separate_process, setup_example_from_tutorials
from gempyor.utils import read_directory


@pytest.mark.skipif(
    os.getenv("FLEPI_PATH") is None,
    reason="The $FLEPI_PATH environment variable is not set.",
)
@pytest.mark.parametrize("config_file", ("config_sample_2pop_vaccine_scenarios.yml",))
@pytest.mark.parametrize("n_jobs", (1, 2))
@pytest.mark.parametrize("start_method", mp.get_all_start_methods())
def test_run_parallel_SEIR_by_multiprocessing_start_method(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    config_file: str,
    n_jobs: int,
    start_method: str,
) -> None:
    """
    Test the parallelization of `run_parallel_SEIR` by multiprocessing start method.

    This test:

    1. Sets up the test environment by copying the necessary files to a temporary
       directory.
    2. Runs a pared down version of `gempyor.simulate.simulate` in a new process.
    3. Reads the contents of the 'spar' directory as a DataFrame.
    4. Tests the contents of the 'spar' DataFrame to ensure that quantities drawn from
       a random distribution are unique across slots.

    The reason for the new process is to control the start method used by
    multiprocessing. The `run_parallel_SEIR` function behaves differently depending on
    the start method used. Under the hood `tqdm.contrib.concurrent.process_map` creates
    a `concurrent.futures.ProcessPoolExecutor` with the default start method (see
    [tqdm/tqdm#1265](https://github.com/tqdm/tqdm/pull/1265)), which is 'spawn' on
    MacOS/Windows and 'fork' on Linux. The work around to this is to force
    multiprocessing to use the desired start method by setting it in the '__main__'
    module with
    [`multiprocessing.set_start_method`](https://docs.python.org/3.11/library/multiprocessing.html#multiprocessing.set_start_method).
    """
    # Test setup
    monkeypatch.chdir(tmp_path)
    setup_example_from_tutorials(tmp_path, config_file)
    with (tmp_path / config_file).open("r") as f:
        config = yaml.safe_load(f)
    nslots = int(config.get("nslots", 1))
    script = Path(__file__).parent.parent / "data" / "run_parallel_test_script.py"

    # Run a pared down version of `gempyor.simulate.simulate` in a new process
    assert (
        run_test_in_separate_process(
            script,
            tmp_path / "test.py",
            args=[str(tmp_path), start_method, str(n_jobs), "false", "true"],
        )
        == 0
    )

    spar = read_directory(tmp_path / "model_output", filters=["spar", ".parquet"])
    spar_successive = read_directory(
        tmp_path / "model_output_successive", filters=["spar", ".parquet"]
    )

    # Test contents of 'spar' DataFrames
    for df in [spar, spar_successive]:
        assert df[df["parameter"] == "Ro"]["value"].nunique() == nslots
    merged_spar = spar.merge(
        spar_successive,
        on=["parameter", "slot"],
        suffixes=("_original", "_successive"),
    )
    assert not merged_spar["value_original"].equals(merged_spar["value_successive"])
