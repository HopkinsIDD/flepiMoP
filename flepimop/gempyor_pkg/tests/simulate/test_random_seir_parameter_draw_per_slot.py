import itertools
import os
from pathlib import Path
from typing import Literal, NamedTuple

from click.testing import CliRunner
import numpy as np
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
    nunique_lower_bound: int | None = None
    exact_value: int | float | None = None
    all_equal_by: str | None = None
    not_all_equal_by: str | None = None

    def assert_df_passes(
        self, dfs: dict[Literal["hnpi", "hpar", "snpi", "spar"], pd.DataFrame]
    ) -> None:
        df = dfs[self.kind].copy()
        for key, value in self.filters.items():
            df = df[df[key] == value]
        if len(df) == 0:
            raise ValueError("No data found for the given filters.")
        if self.nunique is not None:
            assert df[self.column].nunique() == self.nunique
            return
        if self.nunique_lower_bound is not None:
            assert df[self.column].nunique() > self.nunique_lower_bound
            return
        if self.exact_value is not None:
            assert df[self.column].unique() == self.exact_value
            return
        if self.all_equal_by is not None or self.not_all_equal_by is not None:
            expected = False if self.not_all_equal_by else True
            col = self.all_equal_by or self.not_all_equal_by
            column_values = df[col].unique().tolist()
            x = None
            for column_value in column_values:
                y = df[df[col] == column_value][self.column].values
                if x is not None:
                    assert np.array_equal(x, y) is expected
                x = y
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
            "config_sample_2pop_modifiers_test_random.yml",
            None,
            (
                RandomDrawAssertion(
                    kind="hnpi",
                    filters={"subpop": "large_province"},
                    column="value",
                    nunique=10,
                ),
                RandomDrawAssertion(
                    kind="hnpi",
                    filters={"subpop": "small_province"},
                    column="value",
                    nunique=10,
                ),
                RandomDrawAssertion(
                    kind="hpar",
                    filters={
                        "quantity": "probability",
                        "outcome": "incidCase",
                    },
                    column="value",
                    exact_value=0.5,
                ),
                RandomDrawAssertion(
                    kind="hpar",
                    filters={
                        "quantity": "delay",
                        "outcome": "incidCase",
                    },
                    column="value",
                    exact_value=5,
                ),
                RandomDrawAssertion(
                    kind="hpar",
                    filters={
                        "quantity": "probability",
                        "outcome": "incidHosp",
                    },
                    column="value",
                    exact_value=0.05,
                ),
                RandomDrawAssertion(
                    kind="hpar",
                    filters={
                        "quantity": "delay",
                        "outcome": "incidHosp",
                    },
                    column="value",
                    exact_value=7,
                ),
                RandomDrawAssertion(
                    kind="hpar",
                    filters={
                        "quantity": "duration",
                        "outcome": "incidHosp",
                    },
                    column="value",
                    exact_value=10,
                ),
                RandomDrawAssertion(
                    kind="hpar",
                    filters={
                        "quantity": "probability",
                        "outcome": "incidDeath",
                    },
                    column="value",
                    exact_value=0.2,
                ),
                RandomDrawAssertion(
                    kind="hpar",
                    filters={
                        "subpop": "small_province",
                        "quantity": "delay",
                        "outcome": "incidDeath",
                    },
                    column="value",
                    nunique_lower_bound=1,
                ),
                RandomDrawAssertion(
                    kind="hpar",
                    filters={
                        "subpop": "large_province",
                        "quantity": "delay",
                        "outcome": "incidDeath",
                    },
                    column="value",
                    nunique_lower_bound=1,
                ),
                RandomDrawAssertion(
                    kind="hpar",
                    filters={
                        "quantity": "delay",
                        "outcome": "incidDeath",
                    },
                    column="value",
                    all_equal_by="subpop",
                ),
                RandomDrawAssertion(
                    kind="snpi",
                    filters={
                        "modifier_name": "Ro_lockdown",
                    },
                    column="value",
                    exact_value=0.4,
                ),
                RandomDrawAssertion(
                    kind="snpi",
                    filters={
                        "modifier_name": "Ro_relax",
                    },
                    column="value",
                    nunique=20,
                ),
                RandomDrawAssertion(
                    kind="spar",
                    filters={
                        "parameter": "sigma",
                    },
                    column="value",
                    exact_value=0.25,
                ),
                RandomDrawAssertion(
                    kind="spar",
                    filters={
                        "parameter": "gamma",
                    },
                    column="value",
                    exact_value=0.2,
                ),
                RandomDrawAssertion(
                    kind="spar",
                    filters={
                        "parameter": "Ro",
                    },
                    column="value",
                    nunique=10,
                ),
            ),
        ),
        (
            "config_sample_2pop_modifiers_test_random_subpop_groups.yml",
            None,
            (
                RandomDrawAssertion(
                    kind="hnpi",
                    filters={},
                    column="value",
                    nunique=10,
                ),
                RandomDrawAssertion(
                    kind="snpi",
                    filters={"modifier_name": "Ro_lockdown"},
                    column="value",
                    exact_value=0.4,
                ),
                RandomDrawAssertion(
                    kind="snpi",
                    filters={"modifier_name": "Ro_relax"},
                    column="value",
                    nunique=10,
                ),
            ),
        ),
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
                    kind="snpi",
                    filters={},
                    column="value",
                    exact_value=0.0,
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
def test_parameter_draw_per_slot(
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


@pytest.mark.xfail(reason="Parameter matching across scenarios is not yet supported.")
@pytest.mark.skipif(
    os.getenv("FLEPI_PATH") is None,
    reason="The $FLEPI_PATH environment variable is not set.",
)
@pytest.mark.parametrize("n_jobs", [1, 2])
def test_parameter_draws_per_slot_across_scenarios(
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

    # List out scenarios
    scenario_directories = [
        directory
        for directory in (tmp_path / "model_output").iterdir()
        if directory.is_dir()
    ]

    # Check that the parameter draws are the same across scenarios
    for scenario_a, scenario_b in itertools.combinations(scenario_directories, 2):
        for kind in ("hnpi", "hpar", "snpi", "spar"):
            scenario_a_df = read_directory(scenario_a, filters=kind)
            scenario_b_df = read_directory(scenario_b, filters=kind)
            scenario_a_columns = set(scenario_a_df.columns)
            scenario_b_columns = set(scenario_b_df.columns)
            assert scenario_a_columns == scenario_b_columns
            assert "value" in scenario_a_columns
            merge_df = pd.merge(
                scenario_a_df,
                scenario_b_df,
                on=list(scenario_a_columns - {"value"}),
                suffixes=("_a", "_b"),
                copy=True,
            )
            if len(merge_df) == 0:
                # No shared value types to compare
                continue
            value_a = merge_df["value_a"].to_numpy()
            value_b = merge_df["value_b"].to_numpy()
            assert np.allclose(value_a, value_b, equal_nan=True)
            assert np.allclose(value_b, value_a, equal_nan=True)
