import os
from pathlib import Path
from typing import Literal, NamedTuple

from click.testing import CliRunner
import pandas as pd
import pytest

from gempyor.simulate import _click_simulate
from gempyor.testing import setup_example_from_tutorials
from gempyor.utils import read_directory


class RandomDrawAssertion(NamedTuple):
    kind: Literal["hnpi", "hpar", "snpi", "spar"]
    filters: dict[str, str]
    column: str
    nunique: int | None = None
    exact_value: int | float | None = None

    def assert_df_passes(
        self, dfs: dict[Literal["hnpi", "hpar", "snpi", "spar"], pd.DataFrame]
    ) -> None:
        df = dfs[self.kind].copy()
        for key, value in self.filters.items():
            df = df[df[key] == value]
        if self.nunique is not None:
            assert df[self.column].nunique() == self.nunique
            return
        if self.exact_value is not None:
            assert df[self.column].unique() == self.exact_value
            return
        raise ValueError("No assertion was made.")


@pytest.mark.skipif(
    os.getenv("FLEPI_PATH") is None,
    reason="The $FLEPI_PATH environment variable is not set.",
)
@pytest.mark.parametrize("n_jobs", [1, 2])
@pytest.mark.parametrize(
    ("config", "name_filter", "assertions"),
    (
        (
            "config_sample_2pop_vaccine_scenarios.yml",
            "sample_2pop_pess_vax",
            (
                RandomDrawAssertion(
                    kind="hpar",
                    filters={
                        "subpop": "small_province",
                        "quantity": "probability",
                        "outcome": "incidCase",
                    },
                    column="value",
                    nunique=10,
                ),
                RandomDrawAssertion(
                    kind="hpar",
                    filters={
                        "subpop": "large_province",
                        "quantity": "probability",
                        "outcome": "incidCase",
                    },
                    column="value",
                    nunique=10,
                ),
                RandomDrawAssertion(
                    kind="spar",
                    filters={"parameter": "Ro"},
                    column="value",
                    nunique=10,
                ),
            ),
        ),
    ),
)
def test_random_seir_parameter_draw_per_slot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    n_jobs: int,
    config: str,
    name_filter: str | None,
    assertions: tuple[RandomDrawAssertion, ...] | None,
) -> None:
    # Test setup
    monkeypatch.chdir(tmp_path)
    setup_example_from_tutorials(tmp_path, config)

    # Test execution of `gempyor-simulate`
    runner = CliRunner()
    result = runner.invoke(_click_simulate, [config, "--jobs", str(n_jobs)])
    assert result.exit_code == 0

    if assertions:
        # Read in the data to assert on
        filters = [".parquet"]
        if name_filter:
            filters.append(name_filter)
        dfs = {}
        for kind in ("hnpi", "hpar", "snpi", "spar"):
            dfs[kind] = read_directory(
                tmp_path / "model_output",
                filters=[kind] + filters,
            )

        # Test contents of DataFrames
        for assertion in assertions:
            assertion.assert_df_passes(dfs)
