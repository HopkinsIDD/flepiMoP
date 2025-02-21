from collections.abc import Iterable
from datetime import timedelta
import logging
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from confuse import Configuration
import pytest

from gempyor.batch import (
    JobResources,
    JobResult,
    JobSubmission,
    SlurmBatchSystem,
    get_batch_system,
)
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
def test_submit_output_validation(
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
    options: dict[str, Any],
    expected_options: str | None,
    verbosity: int | None,
    dry_run: bool,
) -> None:
    batch_system = get_batch_system("slurm")
    sbatch = str(sample_script(tmp_path, True, name="sbatch").absolute())
    script = sample_script(tmp_path, False, name="run.sbatch")

    with patch("gempyor.batch._shutil_which") as shutil_which_patch:
        shutil_which_patch.return_value = sbatch
        with patch("gempyor.batch.subprocess.run") as subprocess_run_patch:
            mock_process = MagicMock()
            mock_process.returncode = 0
            mock_process.args = [sbatch, str(script.absolute())]
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
                    hash((logging.DEBUG, False)): 4,
                    hash((logging.INFO, True)): 1,
                    hash((logging.INFO, False)): 2,
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


@pytest.mark.parametrize("command", ("echo 'Foobar!'",))
@pytest.mark.parametrize("options", (None, {}, {"job_name": "my_job_name"}))
@pytest.mark.parametrize(
    "verbosity", (None, logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR)
)
@pytest.mark.parametrize("dry_run", (True, False))
def test_submit_command_output_validation(
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    command: str,
    options: dict[str, str | Iterable[str]] | None,
    verbosity: int | None,
    dry_run: bool,
) -> None:
    monkeypatch.chdir(tmp_path)
    batch_system = get_batch_system("slurm")
    with patch.object(batch_system, "submit") as submit_patch:
        submit_patch.return_value = None
        assert batch_system.submit_command(command, options, verbosity, dry_run) is None
        submit_patch.assert_called_once()
        assert len(caplog.records) == (
            0 if verbosity is None else (verbosity <= logging.INFO)
        )
        sbatch_script = submit_patch.call_args.args[0]
        assert str(sbatch_script).endswith(".sbatch")
        if "job_name" in (options or {}):
            assert sbatch_script.name.startswith(options.get("job_name"))
        if dry_run:
            sbatch_script_copy = tmp_path / sbatch_script.name
            assert sbatch_script_copy.exists()
            assert command in sbatch_script_copy.read_text()
        assert len(submit_patch.call_args.args[1]) == (
            0 if options is None else len(options)
        )
        assert submit_patch.call_args.args[2] == verbosity
        assert submit_patch.call_args.args[3] == dry_run


@pytest.mark.parametrize(
    ("submission", "returncode", "stdout", "stderr", "expected"),
    (
        (
            JobSubmission(job_id=123, args=[], returncode=0, stdout="", stderr=""),
            0,
            "\n".join(
                (
                    "Job ID: 123",
                    "Cluster: longleaf",
                    "User/Group: twillard/users",
                    "State: PENDING",
                    "Cores: 1",
                    "Efficiency not available for jobs in the PENDING state.",
                )
            ),
            "",
            JobResult(
                status="pending", returncode=None, wall_time=None, memory_efficiency=None
            ),
        ),
        (
            JobSubmission(job_id=456, args=[], returncode=0, stdout="", stderr=""),
            0,
            "\n".join(
                (
                    "Job ID: 456",
                    "Cluster: longleaf",
                    "User/Group: twillard/users",
                    "State: COMPLETED (exit code 0)",
                    "Cores: 1",
                    "CPU Utilized: 00:00:01",
                    "CPU Efficiency: 50.00% of 00:00:02 core-walltime",
                    "Job Wall-clock time: 00:00:02",
                    "Memory Utilized: 236.00 KB",
                    "Memory Efficiency: 0.02% of 1000.00 MB",
                )
            ),
            "",
            JobResult(
                status="completed",
                returncode=0,
                wall_time=timedelta(seconds=2),
                memory_efficiency=0.0002,
            ),
        ),
        (
            JobSubmission(job_id=456, args=[], returncode=0, stdout="", stderr=""),
            0,
            "\n".join(
                (
                    "Job ID: 456",
                    "Cluster: longleaf",
                    "User/Group: twillard/users",
                    "State: RUNNING",
                    "Cores: 1",
                    "CPU Utilized: 00:00:00",
                    "CPU Efficiency: 0.00% of 00:00:09 core-walltime",
                    "Job Wall-clock time: 00:00:09",
                    "Memory Utilized: 0.00 MB (estimated maximum)",
                    "Memory Efficiency: 0.00% of 500.00 MB (500.00 MB/core)",
                    "WARNING: Efficiency statistics may be misleading for RUNNING jobs.",
                )
            ),
            "",
            JobResult(
                status="running",
                returncode=None,
                wall_time=timedelta(seconds=9),
                memory_efficiency=0.0,
            ),
        ),
        (
            JobSubmission(job_id=789, args=[], returncode=0, stdout="", stderr=""),
            0,
            "\n".join(
                (
                    "Job ID: 59241870",
                    "Cluster: longleaf",
                    "User/Group: twillard/users",
                    "State: FAILED (exit code 1)",
                    "Cores: 1",
                    "CPU Utilized: 00:00:01",
                    "CPU Efficiency: 0.79% of 00:02:07 core-walltime",
                    "Job Wall-clock time: 00:02:07",
                    "Memory Utilized: 656.00 KB",
                    "Memory Efficiency: 0.13% of 500.00 MB",
                )
            ),
            "",
            JobResult(
                status="failed",
                returncode=1,
                wall_time=timedelta(seconds=7, minutes=2),
                memory_efficiency=0.0013,
            ),
        ),
    ),
)
def test_status_output_validation(
    submission: JobSubmission,
    returncode: int,
    stdout: str,
    stderr: str,
    expected: JobResult,
) -> None:
    batch_system = get_batch_system("slurm")
    with patch("gempyor.batch._shutil_which") as shutil_which_patch:
        shutil_which_patch.return_value = "seff"
        with patch("gempyor.batch.subprocess.run") as subprocess_run_patch:
            mock_process = MagicMock()
            mock_process.returncode = returncode
            mock_process.stdout = stdout
            mock_process.stderr = stderr
            subprocess_run_patch.return_value = mock_process
            assert isinstance(job_result := batch_system.status(submission), JobResult)
            assert job_result == expected


@pytest.mark.parametrize(
    "cli_options",
    (
        {},
        {"extra": {}},
        {"extra": {"partition": "foobar"}},
        {"extra": {"email": "janedoe@example.com"}},
        {"extra": {"partition": "fizzbuzz", "email": "jake@statefarm.com"}},
        {"extra": {"partition": "foo", "other_opt": "not relevant"}},
    ),
)
@pytest.mark.parametrize("verbosity", (None, logging.DEBUG, logging.INFO, logging.WARNING))
def test_options_from_config_and_cli(
    caplog: pytest.LogCaptureFixture, cli_options: dict[str, Any], verbosity: int | None
) -> None:
    batch_system = get_batch_system("slurm")
    options = batch_system.options_from_config_and_cli(
        Configuration("foobar", read=True), cli_options, verbosity
    )
    assert isinstance(options, dict)
    assert len(options) == int("partition" in cli_options.get("extra", {})) + 2 * int(
        "email" in cli_options.get("extra", {})
    )
    assert len(caplog.records) == int(verbosity == logging.DEBUG)


@pytest.mark.parametrize(
    ("regex_name", "string", "expected_groups"),
    (
        ("state", "State: PENDING", ("PENDING", None, None)),
        ("state", "state: pending", ("pending", None, None)),
        ("state", "State: COMPLETED (exit code 0)", ("COMPLETED", " (exit code 0)", "0")),
        ("state", "state: completed (exit code 0)", ("completed", " (exit code 0)", "0")),
        ("state", "State: RUNNING", ("RUNNING", None, None)),
        ("state", "state: running", ("running", None, None)),
        ("state", "State: FAILED (exit code 1)", ("FAILED", " (exit code 1)", "1")),
        ("state", "state: failed (exit code 1)", ("failed", " (exit code 1)", "1")),
        ("state", "Cluster: longleaf", None),
        ("state", "cluster: longleaf", None),
        ("state", "Memory Utilized: 656.00 KB", None),
        ("state", "memory utilized: 656.00 KB", None),
        ("state", "CPU Efficiency: 0.79% of 00:02:07 core-walltime", None),
        ("state", "cpu efficiency: 0.79% of 00:02:07 core-walltime", None),
        ("wall_time", "Job Wall-clock time: 00:02:07", ("00", "02", "07")),
        ("wall_time", "job wall-clock time: 00:02:07", ("00", "02", "07")),
        ("wall_time", "Job Wall-clock time: 12:34:56", ("12", "34", "56")),
        ("wall_time", "job wall-clock time: 12:34:56", ("12", "34", "56")),
        ("wall_time", "Efficiency not available for jobs in the PENDING state.", None),
        ("wall_time", "efficiency not available for jobs in the pending state.", None),
        ("wall_time", "User/Group: twillard/users", None),
        ("wall_time", "user/group: twillard/users", None),
        ("memory_efficiency", "Memory Efficiency: 0.02% of 1000.00 MB", ("0.02",)),
        ("memory_efficiency", "memory efficiency: 0.02% of 1000.00 MB", ("0.02",)),
        ("memory_efficiency", "Memory Efficiency: 0.00% of 1000.00 MB", ("0.00",)),
        ("memory_efficiency", "memory efficiency: 0.00% of 1000.00 MB", ("0.00",)),
        ("memory_efficiency", "Memory Efficiency: 0.13% of 500.00 MB", ("0.13",)),
        ("memory_efficiency", "memory efficiency: 0.13% of 500.00 MB", ("0.13",)),
        ("memory_efficiency", "Job Wall-clock time: 00:02:07", None),
        ("memory_efficiency", "job wall-clock time: 00:02:07", None),
        ("memory_efficiency", "User/Group: twillard/users", None),
        ("memory_efficiency", "user/group: twillard/users", None),
        ("memory_efficiency", "CPU Efficiency: 0.79% of 00:02:07 core-walltime", None),
        ("memory_efficiency", "cpu efficiency: 0.79% of 00:02:07 core-walltime", None),
        ("cpu_efficiency", "CPU Efficiency: 0.79% of 00:02:07 core-walltime", ("0.79",)),
        ("cpu_efficiency", "cpu efficiency: 0.79% of 00:02:07 core-walltime", ("0.79",)),
        ("cpu_efficiency", "CPU Efficiency: 0.00% of 00:02:07 core-walltime", ("0.00",)),
        ("cpu_efficiency", "cpu efficiency: 0.00% of 00:02:07 core-walltime", ("0.00",)),
        ("cpu_efficiency", "CPU Efficiency: 50.00% of 00:00:02 core-walltime", ("50.00",)),
        ("cpu_efficiency", "cpu efficiency: 50.00% of 00:00:02 core-walltime", ("50.00",)),
        ("cpu_efficiency", "Job Wall-clock time: 00:02:07", None),
        ("cpu_efficiency", "job wall-clock time: 00:02:07", None),
        ("cpu_efficiency", "Job ID: 59241870", None),
        ("cpu_efficiency", "job id: 59241870", None),
        ("cpu_efficiency", "User/Group: twillard/users", None),
        ("cpu_efficiency", "user/group: twillard/users", None),
        ("cpu_efficiency", "Memory Efficiency: 0.13% of 500.00 MB", None),
        ("cpu_efficiency", "memory efficiency: 0.13% of 500.00 MB", None),
    ),
)
def test_seff_regexes(
    regex_name: str, string: str, expected_groups: tuple[str, ...] | None
) -> None:
    batch_system = get_batch_system("slurm")
    regex = getattr(batch_system, f"_seff_{regex_name}_regex")
    match_groups = m.groups() if (m := regex.match(string)) is not None else None
    assert match_groups == expected_groups
