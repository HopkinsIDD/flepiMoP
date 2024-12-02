import os
from pathlib import Path
import shutil

from click.testing import CliRunner
import pandas as pd
import pytest

from gempyor.simulate import _click_simulate


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


def test_random_seir_parameter_draw_per_slot(
    monkeypatch: pytest.MonkeyPatch, setup_sample_2pop_vaccine_scenarios: Path
) -> None:
    # Test setup
    monkeypatch.chdir(setup_sample_2pop_vaccine_scenarios)

    # Test execution of `gempyor-simulate`
    runner = CliRunner()
    result = runner.invoke(_click_simulate, ["config_sample_2pop_vaccine_scenarios.yml"])
    assert result.exit_code == 0

    # Get the contents of 'spar' and 'hpar' directories as DataFrames
    spar_directory: Path | None = None
    hpar_directory: Path | None = None
    for p in setup_sample_2pop_vaccine_scenarios.rglob("*"):
        if p.is_dir() and p.name == "spar":
            spar_directory = p
        elif p.is_dir() and p.name == "hpar":
            hpar_directory = p
        if spar_directory is not None and hpar_directory is not None:
            break

    def read_directory(directory: Path) -> list[pd.DataFrame]:
        dfs: list[pd.DataFrame] | pd.DataFrame = []
        for i, f in enumerate(sorted(list(directory.glob("*.parquet")))):
            dfs.append(pd.read_parquet(f))
            dfs[-1]["slot"] = i
        dfs = pd.concat(dfs)
        return dfs

    hpar = read_directory(hpar_directory)
    spar = read_directory(spar_directory)

    # Test contents of 'spar'/'hpar' DataFrames
    assert (
        hpar[
            (hpar["subpop"] == "large_province")
            & (hpar["quantity"] == "probability")
            & (hpar["outcome"] == "incidCase")
        ]["value"].nunique()
        == 10
    )
    assert (
        hpar[
            (hpar["subpop"] == "small_province")
            & (hpar["quantity"] == "probability")
            & (hpar["outcome"] == "incidCase")
        ]["value"].nunique()
        == 10
    )
    assert spar[spar["parameter"] == "Ro"]["value"].nunique() == 10
