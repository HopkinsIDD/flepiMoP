from collections.abc import Iterable
import logging
import os
from pathlib import Path
import re
from typing import Literal
from unittest.mock import MagicMock, patch

import pytest

from gempyor.batch import JobSubmission, _submit_via_subprocess
from gempyor.logging import get_script_logger
from gempyor.testing import sample_script
from gempyor.utils import _format_cli_options


def test_exec_does_not_exist_or_is_not_a_file_value_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)

    does_not_exist = tmp_path / "nonexistent"
    assert not does_not_exist.exists()
    with pytest.raises(
        ValueError,
        match=(
            f"^The executable '{does_not_exist.absolute()}' "
            "either does not exist or is not a file.$"
        ),
    ):
        _submit_via_subprocess(does_not_exist, False, "popen", None, None, None, None, True)

    not_a_file = tmp_path / "not_a_file"
    not_a_file.mkdir(parents=True, exist_ok=True)
    assert not_a_file.is_dir()
    with pytest.raises(
        ValueError,
        match=(
            f"^The executable '{not_a_file.absolute()}' "
            "either does not exist or is not a file.$"
        ),
    ):
        _submit_via_subprocess(not_a_file, False, "popen", None, None, None, None, True)


@pytest.mark.parametrize("exec_name", ("example",))
@pytest.mark.parametrize("exec_is_executable", (True, False))
@pytest.mark.parametrize("coerce_exec", (True, False))
@pytest.mark.parametrize("exec_method", ("popen", "run"))
@pytest.mark.parametrize("options", (None, {"output": "/abc/def/ghi.log", "ntasks": "2"}))
@pytest.mark.parametrize("args", (None, ("/path/to/script",)))
@pytest.mark.parametrize("use_job_id_callback", (True, False))
@pytest.mark.parametrize("verbosity", (None, logging.DEBUG, logging.INFO, logging.WARNING))
@pytest.mark.parametrize("dry_run", (True, False))
@pytest.mark.parametrize("returncode", (0, 1))
@pytest.mark.parametrize("stdout", ("Foobar", ""))
@pytest.mark.parametrize("stderr", ("Error", ""))
def test_output_validation_for_select_values(
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
    exec_name: str,
    exec_is_executable: bool,
    coerce_exec: bool,
    exec_method: Literal["popen", "run"],
    options: dict[str, str | Iterable[str]] | None,
    args: Iterable[str] | None,
    use_job_id_callback: bool,
    verbosity: int | None,
    dry_run: bool,
    returncode: int,
    stdout: str,
    stderr: str,
) -> None:
    exec = sample_script(exec_name, tmp_path, exec_is_executable)
    logger = None if verbosity is None else get_script_logger(__name__, verbosity)
    job_id_callback = (
        (lambda proc: proc.pid)
        if exec_method == "popen"
        else (lambda proc: int("".join(re.findall(r"\d+", proc.stdout) or ["-1"])))
    )
    with patch("gempyor.batch.subprocess.run") as subprocess_run_patch:
        mock_completed_process = MagicMock()
        mock_completed_process.returncode = returncode
        mock_completed_process.stdout = stdout
        mock_completed_process.stderr = stderr
        subprocess_run_patch.return_value = mock_completed_process
        with patch("gempyor.batch.subprocess.Popen") as subprocess_popen_patch:
            mock_popened_process = MagicMock()
            mock_popened_process.returncode = returncode
            mock_popened_process.pid = 123
            mock_popened_process.communicate.return_value = (
                f"{stdout}\n".encode(),
                f"{stderr}\n".encode(),
            )
            subprocess_popen_patch.return_value = mock_popened_process

            result = _submit_via_subprocess(
                exec,
                coerce_exec,
                exec_method,
                options,
                args,
                job_id_callback if use_job_id_callback else None,
                logger,
                dry_run,
            )
            assert result is None if dry_run else isinstance(result, JobSubmission)

            assert os.access(exec.absolute(), os.X_OK) == (
                True if coerce_exec else exec_is_executable
            )

            call_args = None
            if dry_run:
                subprocess_run_patch.assert_not_called()
                subprocess_popen_patch.assert_not_called()
            elif exec_method == "popen":
                subprocess_run_patch.assert_not_called()
                subprocess_popen_patch.assert_called_once()
                call_args = subprocess_popen_patch.call_args.args[0]
            else:
                subprocess_run_patch.assert_called_once()
                subprocess_popen_patch.assert_not_called()
                call_args = subprocess_run_patch.call_args.args[0]

            if call_args is not None:
                assert len(call_args) == 1 + sum(
                    [1 if isinstance(v, str) else len(v) for v in (options or {}).values()]
                ) + len(args or [])
                assert Path(call_args[0]) == exec.absolute()
                if options:
                    assert call_args[
                        1 : len(call_args) - len(args or [])
                    ] == _format_cli_options(options)
                if args:
                    assert call_args[-len(args) :] == list(args)

            if result is not None:
                job_id = 123 if exec_method == "popen" else -1
                assert result.job_id == (job_id if use_job_id_callback else None)
                assert result.stdout == stdout
                assert result.stderr == stderr
                assert result.returncode == returncode

            # Calculate the number of logs
            log_count = 0
            if verbosity is not None:
                if verbosity <= logging.DEBUG:
                    log_count += 1
                if (
                    coerce_exec and not exec_is_executable
                ) and verbosity <= logging.WARNING:
                    log_count += 1
                if dry_run and verbosity <= logging.INFO:
                    log_count += 1
                elif not dry_run:
                    if verbosity <= logging.INFO:
                        log_count += 1
                    if returncode != 0 and verbosity <= logging.CRITICAL:
                        log_count += 1
                    if verbosity <= logging.DEBUG and stdout:
                        log_count += 1
                    if verbosity <= logging.ERROR and stderr:
                        log_count += 1
                    if use_job_id_callback and verbosity <= logging.INFO:
                        log_count += 1
            assert len(caplog.records) == log_count
