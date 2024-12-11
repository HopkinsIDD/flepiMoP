from pathlib import Path
from typing import Any

from click.testing import CliRunner
import pytest
import yaml

from gempyor.cli import patch


@pytest.mark.parametrize(
    ("data_one", "data_two", "expected_parameters"),
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
            {"gamma": {"value": 3.4}},
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
            {"gamma": {"value": 3.4}},
        ),
    ),
)
def test_patch_seir_parameters_behavior(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    data_one: dict[str, Any],
    data_two: dict[str, Any],
    expected_parameters: dict[str, Any],
) -> None:
    # Setup the test
    monkeypatch.chdir(tmp_path)
    config_one = tmp_path / "config_one.yml"
    config_one.write_text(yaml.dump(data_one))
    config_two = tmp_path / "config_two.yml"
    config_two.write_text(yaml.dump(data_two))

    # Invoke the command
    runner = CliRunner()
    with pytest.warns(
        UserWarning, match="^Configuration files contain overlapping keys: {'seir'}.$"
    ):
        result = runner.invoke(patch, [config_one.name, config_two.name])
    assert result.exit_code == 0

    # Check the output
    patched_config = yaml.safe_load(result.output)
    assert "seir" in patched_config
    assert patched_config["seir"]["parameters"] == expected_parameters


@pytest.mark.parametrize("indent", (2, 4, 6))
def test_user_provided_indent_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, indent: int
) -> None:
    # Setup the test
    monkeypatch.chdir(tmp_path)
    config = tmp_path / "config.yml"
    config.write_text(yaml.dump({"seir": {"parameters": {"beta": {"value": 1.2}}}}))

    # Invoke the command
    runner = CliRunner()
    result = runner.invoke(patch, [config.name, "--indent", str(indent)])
    assert result.exit_code == 0

    # Check the output indentation, manually since `yaml.load` abstracts spacing away
    for line in result.output.split():
        stripped_line = line.lstrip()
        if stripped_line and not stripped_line.startswith("#"):
            indent_size = len(line) - len(stripped_line)
            if indent_size > 0:
                assert indent_size == indent
                break
