import os
import pytest
import subprocess
from pathlib import Path

from click.testing import CliRunner

from gempyor.simulate import _click_simulate
from gempyor.testing import *
from gempyor.shared_cli import parse_config_files
from gempyor.cli import patch

# See here to test click application https://click.palletsprojects.com/en/8.1.x/testing/
# would be useful to also call the command directly

tutorialpath = os.path.dirname(__file__) + "/../../../../examples/tutorials"


@pytest.mark.slow
def test_config_sample_2pop():
    os.chdir(tutorialpath)
    runner = CliRunner()
    result = runner.invoke(_click_simulate, ["config_sample_2pop.yml"])
    assert result.exit_code == 0


@pytest.mark.slow
def test_config_sample_2pop_deprecated():
    os.chdir(tutorialpath)
    runner = CliRunner()
    result = runner.invoke(_click_simulate, ["-c", "config_sample_2pop.yml"])
    assert result.exit_code == 0


@pytest.mark.slow
def test_sample_2pop_modifiers():
    os.chdir(tutorialpath)
    runner = CliRunner()
    result = runner.invoke(
        _click_simulate,
        [
            "config_sample_2pop.yml",
            "config_sample_2pop_outcomes_part.yml",
            "config_sample_2pop_modifiers_part.yml",
        ],
    )
    assert result.exit_code == 0


def test_sample_2pop_modifiers_combined(tmp_path: Path):
    os.chdir(tutorialpath)
    tmp_cfg1 = tmp_path / "patch_modifiers.yml"
    tmp_cfg2 = tmp_path / "nopatch_modifiers.yml"
    runner = CliRunner()

    result = runner.invoke(
        patch,
        [
            "config_sample_2pop.yml",
            "config_sample_2pop_outcomes_part.yml",
            "config_sample_2pop_modifiers_part.yml",
        ],
    )
    assert result.exit_code == 0
    with open(tmp_cfg1, "w") as f:
        f.write(result.output)

    result = runner.invoke(patch, ["config_sample_2pop_modifiers.yml"])
    assert result.exit_code == 0
    with open(tmp_cfg2, "w") as f:
        f.write(result.output)

    tmpconfig1 = create_confuse_config_from_file(str(tmp_cfg1)).flatten()
    tmpconfig2 = create_confuse_config_from_file(str(tmp_cfg2)).flatten()

    assert {k: v for k, v in tmpconfig1.items() if k != "config_src"} == {
        k: v for k, v in tmpconfig2.items() if k != "config_src"
    }


@pytest.mark.slow
def test_simple_usa_statelevel_more_deprecated():
    os.chdir(tutorialpath + "/../simple_usa_statelevel")
    result = subprocess.run(
        ["gempyor-simulate", "-n", "1", "-c", "simple_usa_statelevel.yml"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
