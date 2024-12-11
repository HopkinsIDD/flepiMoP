from pathlib import Path
from typing import Any

from click.testing import CliRunner
import pytest
import yaml

from gempyor.cli import patch


@pytest.mark.parametrize(
    ("data_one", "data_two"),
    (
        (
            {
                "seir": {
                    "parameters": {
                        "beta": {"value": 1.2},
                    }
                }
            },
            {
                "seir": {
                    "parameters": {
                        "gamma": {"value": 3.4},
                    }
                }
            },
        ),
        (
            {
                "seir": {
                    "parameters": {
                        "sigma": {"value": 5.6},
                        "gamma": {"value": 7.8},
                    }
                }
            },
            {
                "seir": {
                    "parameters": {
                        "gamma": {"value": 3.4},
                    }
                }
            },
        ),
    ),
)
def test_overlapping_sections_value_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    data_one: dict[str, Any],
    data_two: dict[str, Any],
) -> None:
    # Setup the test
    monkeypatch.chdir(tmp_path)
    config_one = tmp_path / "config_one.yml"
    config_one.write_text(yaml.dump(data_one))
    config_two = tmp_path / "config_two.yml"
    config_two.write_text(yaml.dump(data_two))

    # Invoke the command
    runner = CliRunner()
    result = runner.invoke(patch, [config_one.name, config_two.name])
    assert result.exit_code == 1
    assert isinstance(result.exception, ValueError)
    assert str(result.exception) == (
        "Configuration files contain overlapping keys, seir, introduced by config_two.yml."
    )


@pytest.mark.parametrize(
    ("data", "seir_modifier_scenarios", "outcome_modifier_scenarios"),
    (
        (
            {
                "seir_modifiers": {
                    "scenarios": ["Ro_lockdown", "Ro_all"],
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
            },
            [],
            [],
        ),
        (
            {
                "seir_modifiers": {
                    "scenarios": ["Ro_lockdown", "Ro_all"],
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
            },
            ["Ro_all"],
            [],
        ),
        (
            {
                "seir_modifiers": {
                    "scenarios": ["Ro_lockdown", "Ro_all"],
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
            },
            ["Ro_all", "Ro_relax", "Ro_lockdown"],
            [],
        ),
        (
            {
                "outcome_modifiers": {
                    "scenarios": ["test_limits"],
                    "modifiers": {
                        "test_limits": {
                            "method": "SinglePeriodModifier",
                            "parameter": "incidCase::probability",
                            "subpop": "all",
                            "period_start_date": "2020-02-01",
                            "period_end_date": "2020-06-01",
                            "value": 0.5,
                        },
                        "test_expansion": {
                            "method": "SinglePeriodModifier",
                            "parameter": "incidCase::probability",
                            "period_start_date": "2020-06-01",
                            "period_end_date": "2020-08-01",
                            "subpop": "all",
                            "value": 0.7,
                        },
                        "test_limits_expansion": {
                            "method": "StackedModifier",
                            "modifiers": ["test_limits", "test_expansion"],
                        },
                    },
                },
            },
            [],
            [],
        ),
        (
            {
                "outcome_modifiers": {
                    "scenarios": ["test_limits"],
                    "modifiers": {
                        "test_limits": {
                            "method": "SinglePeriodModifier",
                            "parameter": "incidCase::probability",
                            "subpop": "all",
                            "period_start_date": "2020-02-01",
                            "period_end_date": "2020-06-01",
                            "value": 0.5,
                        },
                        "test_expansion": {
                            "method": "SinglePeriodModifier",
                            "parameter": "incidCase::probability",
                            "period_start_date": "2020-06-01",
                            "period_end_date": "2020-08-01",
                            "subpop": "all",
                            "value": 0.7,
                        },
                        "test_limits_expansion": {
                            "method": "StackedModifier",
                            "modifiers": ["test_limits", "test_expansion"],
                        },
                    },
                },
            },
            [],
            ["test_limits_expansion"],
        ),
        (
            {
                "outcome_modifiers": {
                    "scenarios": ["test_limits"],
                    "modifiers": {
                        "test_limits": {
                            "method": "SinglePeriodModifier",
                            "parameter": "incidCase::probability",
                            "subpop": "all",
                            "period_start_date": "2020-02-01",
                            "period_end_date": "2020-06-01",
                            "value": 0.5,
                        },
                        "test_expansion": {
                            "method": "SinglePeriodModifier",
                            "parameter": "incidCase::probability",
                            "period_start_date": "2020-06-01",
                            "period_end_date": "2020-08-01",
                            "subpop": "all",
                            "value": 0.7,
                        },
                        "test_limits_expansion": {
                            "method": "StackedModifier",
                            "modifiers": ["test_limits", "test_expansion"],
                        },
                    },
                },
            },
            [],
            ["test_limits", "test_expansion", "test_limits_expansion"],
        ),
        (
            {
                "seir_modifiers": {
                    "scenarios": ["Ro_lockdown", "Ro_all"],
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
                "outcome_modifiers": {
                    "scenarios": ["test_limits"],
                    "modifiers": {
                        "test_limits": {
                            "method": "SinglePeriodModifier",
                            "parameter": "incidCase::probability",
                            "subpop": "all",
                            "period_start_date": "2020-02-01",
                            "period_end_date": "2020-06-01",
                            "value": 0.5,
                        },
                        "test_expansion": {
                            "method": "SinglePeriodModifier",
                            "parameter": "incidCase::probability",
                            "period_start_date": "2020-06-01",
                            "period_end_date": "2020-08-01",
                            "subpop": "all",
                            "value": 0.7,
                        },
                        "test_limits_expansion": {
                            "method": "StackedModifier",
                            "modifiers": ["test_limits", "test_expansion"],
                        },
                    },
                },
            },
            [],
            [],
        ),
        (
            {
                "seir_modifiers": {
                    "scenarios": ["Ro_lockdown", "Ro_all"],
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
                "outcome_modifiers": {
                    "scenarios": ["test_limits"],
                    "modifiers": {
                        "test_limits": {
                            "method": "SinglePeriodModifier",
                            "parameter": "incidCase::probability",
                            "subpop": "all",
                            "period_start_date": "2020-02-01",
                            "period_end_date": "2020-06-01",
                            "value": 0.5,
                        },
                        "test_expansion": {
                            "method": "SinglePeriodModifier",
                            "parameter": "incidCase::probability",
                            "period_start_date": "2020-06-01",
                            "period_end_date": "2020-08-01",
                            "subpop": "all",
                            "value": 0.7,
                        },
                        "test_limits_expansion": {
                            "method": "StackedModifier",
                            "modifiers": ["test_limits", "test_expansion"],
                        },
                    },
                },
            },
            ["Ro_relax"],
            ["test_expansion"],
        ),
    ),
)
def test_editing_modifier_scenarios(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    data: dict[str, Any],
    seir_modifier_scenarios: list[str],
    outcome_modifier_scenarios: list[str],
) -> None:
    # Setup the test
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / "config.yml"
    config_path.write_text(yaml.dump(data))

    # Invoke the command
    runner = CliRunner()
    args = [config_path.name]
    if seir_modifier_scenarios:
        for s in seir_modifier_scenarios:
            args += ["--seir_modifiers_scenarios", s]
    if outcome_modifier_scenarios:
        for o in outcome_modifier_scenarios:
            args += ["--outcome_modifiers_scenarios", o]
    result = runner.invoke(patch, args)
    assert result.exit_code == 0

    # Check the output
    patched_data = yaml.safe_load(result.output)
    assert "seir_modifiers_scenarios" not in patched_data
    assert patched_data.get("seir_modifiers", {}).get("scenarios", []) == (
        seir_modifier_scenarios
        if seir_modifier_scenarios
        else data.get("seir_modifiers", {}).get("scenarios", [])
    )
    assert "outcome_modifiers_scenarios" not in patched_data
    assert patched_data.get("outcome_modifiers", {}).get("scenarios", []) == (
        outcome_modifier_scenarios
        if outcome_modifier_scenarios
        else data.get("outcome_modifiers", {}).get("scenarios", [])
    )
