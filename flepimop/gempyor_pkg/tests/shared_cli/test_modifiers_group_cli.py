from pathlib import Path
import re
from typing import NamedTuple

from click.testing import CliRunner
import pandas as pd
import pytest
import yaml

from gempyor.NPI.base import config_plot


class SampleInput(NamedTuple):
    config: Path
    project_path: Path
    subpops: list[str]


@pytest.fixture
def config_with_modifiers(tmp_path: Path) -> SampleInput:
    for file, contents in (
        (
            "config_with_modifiers.yml",
            {
                "name": "sample_2pop",
                "setup_name": "minimal",
                "start_date": "2020-02-01",
                "end_date": "2020-08-31",
                "nslots": 1,
                "subpop_setup": {
                    "geodata": "model_input/geodata_sample_2pop.csv",
                    "mobility": "model_input/mobility_sample_2pop.csv",
                },
                "initial_conditions": {
                    "method": "SetInitialConditions",
                    "initial_conditions_file": "model_input/ic_2pop.csv",
                    "allow_missing_subpops": True,
                    "allow_missing_compartments": True,
                },
                "compartments": {"infection_stage": ["S", "E", "I", "R"]},
                "seir": {
                    "integration": {"method": "rk4", "dt": 1},
                    "parameters": {
                        "sigma": {"value": 1 / 4},
                        "gamma": {"value": 1 / 5},
                        "Ro": {"value": 2.5},
                    },
                    "transitions": [
                        {
                            "source": ["S"],
                            "destination": ["E"],
                            "rate": ["Ro * gamma"],
                            "proportional_to": [["S"], ["I"]],
                            "proportion_exponent": ["1", "1"],
                        },
                        {
                            "source": ["E"],
                            "destination": ["I"],
                            "rate": ["sigma"],
                            "proportional_to": ["E"],
                            "proportion_exponent": ["1"],
                        },
                        {
                            "source": ["I"],
                            "destination": ["R"],
                            "rate": ["gamma"],
                            "proportional_to": ["I"],
                            "proportion_exponent": ["1"],
                        },
                    ],
                },
                "seir_modifiers": {
                    "scenarios": ["Ro_all"],
                    "modifiers": {
                        "Ro_lockdown": {
                            "method": "SinglePeriodModifier",
                            "parameter": "Ro",
                            "period_start_date": "2020-03-15",
                            "period_end_date": "2020-05-01",
                            "subpop": "all",
                            "value": 0.4,
                        },
                        "Ro_relax": {
                            "method": "SinglePeriodModifier",
                            "parameter": "Ro",
                            "period_start_date": "2020-05-01",
                            "period_end_date": "2020-07-01",
                            "subpop": "all",
                            "value": 0.8,
                        },
                        "Ro_all": {
                            "method": "StackedModifier",
                            "modifiers": ["Ro_lockdown", "Ro_relax"],
                        },
                    },
                },
                "outcomes": {
                    "method": "delayframe",
                    "outcomes": {
                        "incidCase": {
                            "source": {"incidence": {"infection_stage": "I"}},
                            "probability": {"value": 0.5},
                            "delay": {"value": 5},
                        },
                        "incidHosp": {
                            "source": {"incidence": {"infection_stage": "I"}},
                            "probability": {"value": 0.05},
                            "delay": {"value": 7},
                            "duration": {"value": 10, "name": "currHosp"},
                        },
                        "incidDeath": {
                            "source": "incidHosp",
                            "probability": {"value": 0.2},
                            "delay": {"value": 14},
                        },
                    },
                },
                "outcome_modifiers": {
                    "scenarios": ["test_limits"],
                    "modifiers": {
                        "test_limits": {
                            "method": "SinglePeriodModifier",
                            "parameter": "incidCase",
                            "subpop": "all",
                            "period_start_date": "2020-02-01",
                            "period_end_date": "2020-06-01",
                            "value": 0.5,
                        },
                    },
                },
            },
        ),
        (
            "model_input/geodata_sample_2pop.csv",
            pd.DataFrame.from_records(
                [
                    {"subpop": "large_province", "population": 1000.0},
                    {"subpop": "small_province", "population": 100.0},
                ]
            ),
        ),
        (
            "model_input/mobility_sample_2pop.csv",
            pd.DataFrame.from_records(
                [
                    {"ori": "large_province", "dest": "small_province", "amount": 10.0},
                    {"ori": "small_province", "dest": "large_province", "amount": 1.0},
                ]
            ),
        ),
        (
            "model_input/ic_2pop.csv",
            pd.DataFrame.from_records(
                [
                    {"subpop": "large_province", "mc_name": "S", "amount": 1000.0 - 25.0},
                    {"subpop": "large_province", "mc_name": "E", "amount": 20.0},
                    {"subpop": "large_province", "mc_name": "I", "amount": 5.0},
                    {"subpop": "small_province", "mc_name": "S", "amount": 100.0 - 2.0},
                    {"subpop": "small_province", "mc_name": "E", "amount": 1.0},
                    {"subpop": "small_province", "mc_name": "I", "amount": 1.0},
                ]
            ),
        ),
    ):
        p = tmp_path / file
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open(mode="w") as f:
            if isinstance(contents, pd.DataFrame):
                contents.to_csv(f, index=True)
            else:
                yaml.dump(contents, stream=f)
    return SampleInput(
        config=tmp_path / "config_with_modifiers.yml",
        project_path=tmp_path,
        subpops=["large_province", "small_province"],
    )


def test_config_plot_output(
    monkeypatch: pytest.MonkeyPatch, config_with_modifiers: SampleInput
) -> None:
    # Setup
    monkeypatch.chdir(config_with_modifiers.project_path)

    # Execute CLI
    runner = CliRunner()
    result = runner.invoke(
        config_plot,
        [
            "--config",
            config_with_modifiers.config.name,
            "--project_path",
            str(config_with_modifiers.project_path.absolute()),
        ],
    )
    assert result.exit_code == 0
    assert len(result.output) > 0
    assert result.return_value is None

    # Check output
    plot_pdfs = list(config_with_modifiers.project_path.glob("*.pdf"))
    assert len(plot_pdfs) == 2 + (2 * len(config_with_modifiers.subpops))
    assert (config_with_modifiers.project_path / "outcomesNPIcaveat.pdf").exists()
    for subpop in config_with_modifiers.subpops:
        for modifier_type in ("outcomes", "seir"):
            assert (
                config_with_modifiers.project_path
                / f"{modifier_type}_modifiers_activation_{subpop}.pdf"
            ).exists()
    assert (
        sum(
            re.match(r"^unique\_parsed\_parameters\_.*\.pdf$", p.name) is not None
            for p in plot_pdfs
        )
        == 1
    )
