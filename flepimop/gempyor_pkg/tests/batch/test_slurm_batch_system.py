from datetime import timedelta
import logging
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from gempyor.batch import JobResources, JobSubmission, SlurmBatchSystem, get_batch_system
from gempyor.logging import _get_logging_level
from gempyor.testing import sample_script


def test_slurm_batch_system_registered_by_default() -> None:
    batch_system = get_batch_system("slurm")
    assert isinstance(batch_system, SlurmBatchSystem)
    assert batch_system.name == "slurm"


@pytest.mark.parametrize(
    ("job_resources", "expected"),
    (
        (JobResources(nodes=1, cpus=1, memory=1), "1MB"),
        (JobResources(nodes=1, cpus=1, memory=1024), "1024MB"),
    ),
)
def test_memory_formatting_for_select_values(
    job_resources: JobResources, expected: str
) -> None:
    batch_system = get_batch_system("slurm")
    assert batch_system.format_memory(job_resources) == expected


@pytest.mark.parametrize(
    ("job_time_limit", "expected"),
    (
        (timedelta(hours=1), "1:00:00"),
        (timedelta(hours=1, minutes=30), "1:30:00"),
        (timedelta(days=1), "24:00:00"),
        (timedelta(days=1, hours=2, minutes=34, seconds=56), "26:34:56"),
    ),
)
def test_time_limit_formatting_for_select_values(
    job_time_limit: timedelta, expected: str
) -> None:
    batch_system = get_batch_system("slurm")
    assert batch_system.format_time_limit(job_time_limit) == expected


@pytest.mark.parametrize(
    ("options", "expected_options"),
    (
        ({}, None),
        ({"time": "1:00:00", "mem": "2GB"}, "--time=1:00:00 --mem=2GB"),
        (
            {"output": "/abc/def/ghi.log", "ntasks": "2"},
            "--output=/abc/def/ghi.log --ntasks=2",
        ),
        (
            {"J": "My Job Name", "e": "/path/to/error.log"},
            "-J='My Job Name' -e=/path/to/error.log",
        ),
    ),
)
@pytest.mark.parametrize("verbosity", (None, logging.DEBUG, logging.INFO, logging.WARNING))
@pytest.mark.parametrize("dry_run", (True, False))
def test_output_validation(
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
    options: dict[str, Any],
    expected_options: str | None,
    verbosity: int | None,
    dry_run: bool,
) -> None:
    batch_system = get_batch_system("slurm")
    script = sample_script(tmp_path, False)

    with patch("gempyor.batch._shutil_which") as shutil_which_patch:
        shutil_which_patch.return_value = "sbatch"
        with patch("gempyor.batch.subprocess.run") as subprocess_run_patch:
            mock_process = MagicMock()
            mock_process.returncode = 0
            mock_process.args = ["sbatch", str(script.absolute())]
            mock_process.stdout = "Submitted batch job 999"
            mock_process.stderr = ""
            subprocess_run_patch.return_value = mock_process

            job_result = batch_system.submit(script, options, verbosity, dry_run)
            assert job_result is None if dry_run else isinstance(job_result, JobSubmission)

            shutil_which_patch.assert_called_once_with("sbatch")
            if dry_run:
                subprocess_run_patch.assert_not_called()
            else:
                subprocess_run_patch.assert_called_once()

            if verbosity is None:
                assert len(caplog.records) == 0
            else:
                log_messages_by_level = {
                    hash((logging.DEBUG, True)): 2,
                    hash((logging.DEBUG, False)): 3,
                    hash((logging.INFO, True)): 1,
                    hash((logging.INFO, False)): 1,
                }
                assert len(caplog.records) == log_messages_by_level.get(
                    hash((_get_logging_level(verbosity), dry_run)), 0
                )

            if not dry_run:
                args: list[str] = subprocess_run_patch.call_args.args[0]
                assert len(args) == 2 + len(options)
                assert Path(args[-1]) == script.absolute()
                if options:
                    assert " ".join(args[1:-1]) == expected_options
