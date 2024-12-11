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
