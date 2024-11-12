import os
from pathlib import Path
from typing import Any

from click.testing import CliRunner
import pytest
import yaml

from gempyor.batch import _click_submit


def test_batch_system_aws_not_implemented_error(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yml"
    with config_file.open(mode="w") as f:
        yaml.dump({"name": "foobar", "inference": {"method": "emcee"}}, f)

    runner = CliRunner()
    result = runner.invoke(
        _click_submit,
        [
            "--aws",
            "--simulations",
            "1",
            "--jobs",
            "1",
            "--blocks",
            "1",
            str(config_file.absolute()),
        ],
    )

    assert result.exit_code == 1
    assert isinstance(result.exception, NotImplementedError)
    assert (
        str(result.exception)
        == "The `flepimop submit` CLI does not support batch submission to AWS yet."
    )


def test_cluster_required_for_slurm_value_error(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yml"
    with config_file.open(mode="w") as f:
        yaml.dump({"name": "foobar", "inference": {"method": "emcee"}}, f)

    runner = CliRunner()
    result = runner.invoke(
        _click_submit,
        [
            "--slurm",
            "--simulations",
            "1",
            "--jobs",
            "1",
            "--blocks",
            "1",
            str(config_file.absolute()),
        ],
    )

    assert result.exit_code == 1
    assert isinstance(result.exception, ValueError)
    assert (
        str(result.exception)
        == "When submitting a batch job to slurm a cluster is required."
    )
