"""
Unit tests for parameter draws made by the `flepimop simulate` command.

This test file contains two tests:

1. `test_parameter_draw_per_slot`: This test checks the parameter draws made by the
   `flepimop simulate` command for a specific configuration file. It verifies that the
   parameter draws are as expected for different scenarios and subpopulations. The test
   uses the `RandomDrawAssertion` class to define the expected behavior of the
   parameter draws.
2. `test_parameter_draws_per_slot_across_scenarios`: This test checks that the parameter
   draws made by the `flepimop simulate` command across scenarios are consistent. It is
   currently marked as expected to fail because parameter matching across scenarios is
   not yet supported.

The tests are parameterized to run with different configuration files, multiprocessing
start methods, and number of jobs. For the first test in particular we are checking for:

|                             | Varies By Slot | Varies By Location |
|-----------------------------|----------------|--------------------|
| `spar`                      | Yes            | No                 |
| `hpar`                      | Yes            | No                 |
| `snpi`                      | Yes            | Yes                |
| `hnpi`                      | Yes            | Yes                |
| `snpi` with `subpop_groups` | Yes            | No                 |
| `hnpi` with `subpop_groups` | Yes            | No                 |

"""

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
    """
    Represents an assertion to be made on a DataFrame.

    Attributes:
        kind: The kind of DataFrame to assert on (e.g., "hnpi", "hpar", "snpi", "spar").
        filters: A dictionary of filters to apply to the DataFrame.
        column: The column to assert on.
        nunique: The expected number of unique values in the column or `None` to skip
            this assertion.
        nunique_lower_bound: The lower bound for the number of unique values in the
            column or `None` to skip this assertion.
        exact_value: The exact value to assert in the column or `None` to skip this
            assertion.
        all_equal_by: The column by which all values should be equal or `None` to skip
            this assertion.
        not_all_equal_by: The column by which not all values should be equal or `None`
            to skip this assertion.
    """

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
        """
        Asserts that the DataFrame passes the specified assertions.

        Args:
            dfs: A dictionary of DataFrames to assert on, keyed by kind.

        Raises:
            ValueError: If no data is found for the given filters.
            ValueError: If no assertion was made.
        """
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
                # For the hnpi model output filtered on where subpop is 'large_province'
                # we expect there to be 10 unique values in the 'value' column (unique
                # across slots).
                RandomDrawAssertion(
                    kind="hnpi",
                    filters={"subpop": "large_province"},
                    column="value",
                    nunique=10,
                ),
                # For the hnpi model output filtered on where subpop is 'small_province'
                # we expect there to be 10 unique values in the 'value' column (unique
                # across slots).
                RandomDrawAssertion(
                    kind="hnpi",
                    filters={"subpop": "small_province"},
                    column="value",
                    nunique=10,
                ),
                # For the hpar model output filtered on where quantity is 'probability'
                # and outcome is 'incidCase' we expect the value to be 0.5 for all
                # entries.
                RandomDrawAssertion(
                    kind="hpar",
                    filters={
                        "quantity": "probability",
                        "outcome": "incidCase",
                    },
                    column="value",
                    exact_value=0.5,
                ),
                # For the hpar model output filtered on where quantity is 'delay' and
                # outcome is 'incidCase' we expect the value to be 5 for all entries.
                RandomDrawAssertion(
                    kind="hpar",
                    filters={
                        "quantity": "delay",
                        "outcome": "incidCase",
                    },
                    column="value",
                    exact_value=5,
                ),
                # For the hpar model output filtered on where quantity is 'probability'
                # and outcome is 'incidHosp' we expect the value to be 0.05 for all
                # entries.
                RandomDrawAssertion(
                    kind="hpar",
                    filters={
                        "quantity": "probability",
                        "outcome": "incidHosp",
                    },
                    column="value",
                    exact_value=0.05,
                ),
                # For the hpar model output filtered on where quantity is 'delay' and
                # outcome is 'incidHosp' we expect the value to be 7 for all entries.
                RandomDrawAssertion(
                    kind="hpar",
                    filters={
                        "quantity": "delay",
                        "outcome": "incidHosp",
                    },
                    column="value",
                    exact_value=7,
                ),
                # For the hpar model output filtered on where quantity is 'duration' and
                # outcome is 'incidHosp' we expect the value to be 10 for all entries.
                RandomDrawAssertion(
                    kind="hpar",
                    filters={
                        "quantity": "duration",
                        "outcome": "incidHosp",
                    },
                    column="value",
                    exact_value=10,
                ),
                # For the hpar model output filtered on where quantity is 'probability'
                # and outcome is 'incidDeath' we expect the value to be 0.2 for all
                # entries.
                RandomDrawAssertion(
                    kind="hpar",
                    filters={
                        "quantity": "probability",
                        "outcome": "incidDeath",
                    },
                    column="value",
                    exact_value=0.2,
                ),
                # For the hpar model output filtered on where subpop is
                # 'small_province', quantity is 'delay', and outcome is 'incidDeath' we
                # expect the value column to contain more than 1 unique values.
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
                # For the hpar model output filtered on where subpop is
                # 'large_province', quantity is 'delay', and outcome is 'incidDeath' we
                # expect the value column to contain more than 1 unique values.
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
                # For the hpar model output filtered on where quantity is 'delay' and
                # outcome is 'incidDeath' we expect the value column to be the same
                # across subpop.
                RandomDrawAssertion(
                    kind="hpar",
                    filters={
                        "quantity": "delay",
                        "outcome": "incidDeath",
                    },
                    column="value",
                    all_equal_by="subpop",
                ),
                # For the snpi model output filtered on where modifier_name is
                # 'Ro_lockdown' we expect the value column to be 0.4 for all entries.
                RandomDrawAssertion(
                    kind="snpi",
                    filters={
                        "modifier_name": "Ro_lockdown",
                    },
                    column="value",
                    exact_value=0.4,
                ),
                # For the snpi model output filtered on where modifier_name is
                # 'Ro_relax' we expect the value column to contain 20 unique values
                # (unique across slots and subpops).
                RandomDrawAssertion(
                    kind="snpi",
                    filters={
                        "modifier_name": "Ro_relax",
                    },
                    column="value",
                    nunique=20,
                ),
                # For the spar model output filtered on where parameter is 'sigma' we
                # expect the value column to be 0.25 for all entries.
                RandomDrawAssertion(
                    kind="spar",
                    filters={
                        "parameter": "sigma",
                    },
                    column="value",
                    exact_value=0.25,
                ),
                # For the spar model output filtered on where parameter is 'gamma' we
                # expect the value column to be 0.2 for all entries.
                RandomDrawAssertion(
                    kind="spar",
                    filters={
                        "parameter": "gamma",
                    },
                    column="value",
                    exact_value=0.2,
                ),
                # For the spar model output filtered on where parameter is 'Ro' we
                # expect the value column to contain 10 unique values (unique across
                # slots).
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
                # For the hnpi model output with no filters we expect the value column
                # to contain 10 unique values (unique across slots).
                RandomDrawAssertion(
                    kind="hnpi",
                    filters={},
                    column="value",
                    nunique=10,
                ),
                # For the snpi model output filtered on where modifier_name is
                # 'Ro_lockdown' we expect the value column to be 0.4 for all entries.
                RandomDrawAssertion(
                    kind="snpi",
                    filters={"modifier_name": "Ro_lockdown"},
                    column="value",
                    exact_value=0.4,
                ),
                # For the snpi model output filtered on where modifier_name is
                # 'Ro_relax' we expect the value column to contain 10 unique values
                # (unique across slots).
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
                # For the hpar model output filtered on where subpop is
                # 'small_province', quantity is 'probability', and outcome is
                # 'incidCase' we expect the value column to contain 10 unique values
                # (unique across slots).
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
                # For the hpar model output filtered on where subpop is
                # 'large_province', quantity is 'probability', and outcome is
                # 'incidCase' we expect the value column to contain 10 unique values
                # (unique across slots).
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
                # For the snpi model output with no filters we expect the value column
                # to be 0.0 for all entries.
                RandomDrawAssertion(
                    kind="snpi",
                    filters={},
                    column="value",
                    exact_value=0.0,
                ),
                # For the spar model output filtered on where parameter is 'Ro' we
                # expect the value column to contain 10 unique values (unique across
                # slots).
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
    """Unit test for parameter draw per slot produced by simulate CLI."""
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
    """Unit test for parameter draw per slot across scenarios."""
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
