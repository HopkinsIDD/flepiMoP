import os
import subprocess

from click.testing import CliRunner

from gempyor.simulate import _click_simulate

# See here to test click application https://click.palletsprojects.com/en/8.1.x/testing/
# would be useful to also call the command directly


def test_config_sample_2pop():
    os.chdir(os.path.dirname(__file__) + "/tutorials")
    runner = CliRunner()
    result = runner.invoke(_click_simulate, ["config_sample_2pop.yml"])
    print(result.output)  # useful for debug
    print(result.exit_code)  # useful for debug
    print(result.exception)  # useful for debug
    assert result.exit_code == 0
    assert "completed in" in result.output


def test_config_sample_2pop_deprecated():
    os.chdir(os.path.dirname(__file__) + "/tutorials")
    runner = CliRunner()
    result = runner.invoke(_click_simulate, ["-c", "config_sample_2pop.yml"])
    print(result.output)  # useful for debug
    print(result.exit_code)  # useful for debug
    print(result.exception)  # useful for debug
    assert result.exit_code == 0
    assert "completed in" in result.output


def test_sample_2pop_modifiers():
    os.chdir(os.path.dirname(__file__) + "/tutorials")
    runner = CliRunner()
    result = runner.invoke(
        _click_simulate,
        [
            "config_sample_2pop.yml",
            "config_sample_2pop_outcomes_part.yml",
            "config_sample_2pop_modifiers_part.yml",
        ],
    )
    print(result.output)  # useful for debug
    print(result.exit_code)  # useful for debug
    print(result.exception)  # useful for debug
    assert result.exit_code == 0
    assert "completed in" in result.output


def test_sample_2pop_modifiers_combined():
    os.chdir(os.path.dirname(__file__) + "/tutorials")
    runner = CliRunner()
    result = runner.invoke(_click_simulate, ["config_sample_2pop_modifiers.yml"])
    print(result.output)  # useful for debug
    print(result.exit_code)  # useful for debug
    print(result.exception)  # useful for debug
    assert result.exit_code == 0
    assert "completed in" in result.output


def test_sample_2pop_modifiers_combined_deprecated():
    os.chdir(os.path.dirname(__file__) + "/tutorials")
    runner = CliRunner()
    result = runner.invoke(_click_simulate, ["-c", "config_sample_2pop_modifiers.yml"])
    print(result.output)  # useful for debug
    print(result.exit_code)  # useful for debug
    print(result.exception)  # useful for debug
    assert result.exit_code == 0
    assert "completed in" in result.output


def test_simple_usa_statelevel_deprecated():
    os.chdir(os.path.dirname(__file__) + "/simple_usa_statelevel")
    runner = CliRunner()
    result = runner.invoke(_click_simulate, ["-n", "1", "-c", "simple_usa_statelevel.yml"])
    print(result.output)  # useful for debug
    print(result.exit_code)  # useful for debug
    print(result.exception)  # useful for debug
    assert result.exit_code == 0
    assert "completed in" in result.output


def test_simple_usa_statelevel_more_deprecated():
    os.chdir(os.path.dirname(__file__) + "/simple_usa_statelevel")
    result = subprocess.run(
        ["gempyor-simulate", "-n", "1", "-c", "simple_usa_statelevel.yml"],
        capture_output=True,
        text=True,
    )
    print(result.stdout)  # useful for debug
    print(result.stderr)  # useful for debug
    print(result.returncode)  # useful for debug
    assert result.returncode == 0
