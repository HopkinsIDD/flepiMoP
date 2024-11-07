from pathlib import Path
from typing import Any

from click.testing import CliRunner
import pytest
import yaml

from gempyor.batch import _click_batch


@pytest.mark.parametrize(
    "config",
    (
        {},
        {"inference": {"abc": 123}},
        {"inference": {"method": "legacy"}},
    ),
)
def test_only_inference_emcee_supported_not_implemented_error(
    tmp_path: Path, config: dict[str, Any]
) -> None:
    config_file = tmp_path / "config.yml"
    with config_file.open(mode="w") as f:
        yaml.dump(config, f)

    runner = CliRunner()
    result = runner.invoke(_click_batch, [str(config_file.absolute())])

    assert result.exit_code == 1
    assert isinstance(result.exception, NotImplementedError)
    assert (
        str(result.exception)
        == "The `flepimop batch` CLI only supports EMCEE inference jobs."
    )


@pytest.mark.parametrize(
    "args", (["--aws"], ["--batch-system", "aws"], ["--local"], ["--batch-system", "local"])
)
def test_only_slurm_batch_system_supported_not_implemented_error(
    tmp_path: Path, args: list[str]
) -> None:
    config_file = tmp_path / "config.yml"
    with config_file.open(mode="w") as f:
        yaml.dump({"inference": {"method": "emcee"}}, f)

    runner = CliRunner()
    result = runner.invoke(_click_batch, args + [str(config_file.absolute())])

    assert result.exit_code == 1
    assert isinstance(result.exception, NotImplementedError)
    assert (
        str(result.exception)
        == "The `flepimop batch` CLI only supports batch submission to slurm."
    )


def test_cluster_required_for_slurm_value_error(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yml"
    with config_file.open(mode="w") as f:
        yaml.dump({"inference": {"method": "emcee"}}, f)

    runner = CliRunner()
    result = runner.invoke(
        _click_batch,
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
