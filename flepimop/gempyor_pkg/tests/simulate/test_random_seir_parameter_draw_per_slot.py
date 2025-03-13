import os
from pathlib import Path

from click.testing import CliRunner
import pytest

from gempyor.simulate import _click_simulate
from gempyor.testing import setup_example_from_tutorials
from gempyor.utils import read_directory


@pytest.mark.skipif(
    os.getenv("FLEPI_PATH") is None,
    reason="The $FLEPI_PATH environment variable is not set.",
)
@pytest.mark.parametrize("n_jobs", [1, 2])
def test_random_seir_parameter_draw_per_slot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, n_jobs: int
) -> None:
    # Test setup
    monkeypatch.chdir(tmp_path)
    setup_example_from_tutorials(tmp_path, "config_sample_2pop_vaccine_scenarios.yml")

    # Test execution of `gempyor-simulate`
    runner = CliRunner()
    result = runner.invoke(
        _click_simulate, ["config_sample_2pop_vaccine_scenarios.yml", "--jobs", str(n_jobs)]
    )
    assert result.exit_code == 0

    hpar = read_directory(
        tmp_path / "model_output",
        filters=["sample_2pop_pess_vax", "hpar", ".parquet"],
    )
    spar = read_directory(
        tmp_path / "model_output",
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
