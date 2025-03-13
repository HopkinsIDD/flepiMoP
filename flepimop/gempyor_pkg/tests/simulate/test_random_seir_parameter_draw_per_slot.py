import os
from pathlib import Path
import shutil

from click.testing import CliRunner
import pytest

from gempyor.simulate import _click_simulate
from gempyor.utils import read_directory


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


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_random_seir_parameter_draw_per_slot(
    monkeypatch: pytest.MonkeyPatch, setup_sample_2pop_vaccine_scenarios: Path, n_jobs: int
) -> None:
    # Test setup
    monkeypatch.chdir(setup_sample_2pop_vaccine_scenarios)

    # Test execution of `gempyor-simulate`
    runner = CliRunner()
    result = runner.invoke(
        _click_simulate, ["config_sample_2pop_vaccine_scenarios.yml", "--jobs", str(n_jobs)]
    )
    assert result.exit_code == 0

    hpar = read_directory(
        setup_sample_2pop_vaccine_scenarios,
        filters=["sample_2pop_pess_vax", "hpar", ".parquet"],
    )
    spar = read_directory(
        setup_sample_2pop_vaccine_scenarios,
        filters=["sample_2pop_pess_vax", "spar", ".parquet"],
    )

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
