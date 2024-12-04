import logging
import os
from pathlib import Path
from unittest.mock import patch
import subprocess
from typing import Any, Literal

import pytest

from gempyor.batch import _sbatch
from gempyor.logging import _get_logging_level
from gempyor.utils import _shutil_which


@pytest.fixture
def sample_sbatch_script(tmp_path: Path) -> Path:
    script = tmp_path / "example.sbatch"
    script.write_text("#!/usr/bin/env bash\necho 'Hello sbatch!'")
    return script


def test_export_option_causes_value_error(sample_sbatch_script: Path) -> None:
    with pytest.raises(
        ValueError,
        match="^Found 'export' in `options`, please use `environment_variables` instead.$",
    ):
        _sbatch(sample_sbatch_script, {}, {"export": "VAR1=1,VAR2=abc"}, None, True)


@pytest.mark.parametrize(
    ("environment_variables", "expected_export"),
    (
        ({}, None),
        ({"ENV_VAR": "true"}, "ENV_VAR=true"),
        ({"VAR1": 1, "VAR2": "abc", "VAR3": True}, "VAR1=1,VAR2=abc,VAR3=True"),
        (
            {"NEEDS_ESCAPING": "`", "ALSO_ESCAPED": "a b c"},
            "NEEDS_ESCAPING='`',ALSO_ESCAPED='a b c'",
        ),
        ("all", "ALL"),
        ("nil", "NIL"),
        ("none", "NONE"),
    ),
)
@pytest.mark.parametrize(
    ("options", "expected_options"),
    (
        ({}, None),
        ({"time": "1:00:00", "mem": "2GB"}, "--time=1:00:00 --mem=2GB"),
        (
            {"output": Path("/abc/def/ghi.log"), "ntasks": 2},
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
    sample_sbatch_script: Path,
    environment_variables: dict[str, Any] | Literal["all", "nil", "none"],
    expected_export: str | None,
    options: dict[str, Any],
    expected_options: str | None,
    verbosity: int | None,
    dry_run: bool,
) -> None:
    def shutil_which_wraps(
        cmd: str,
        mode: int = os.F_OK | os.X_OK,
        path: str | bytes | os.PathLike | None = None,
        check: bool = True,
    ) -> str | None:
        return (
            "sbatch"
            if cmd == "sbatch"
            else _shutil_which(cmd, mode=mode, path=path, check=check)
        )

    def subprocess_run_wraps(args, **kwargs):
        if os.path.basename(args[0]) == "sbatch":
            return subprocess.CompletedProcess(
                args=args, returncode=0, stdout=b"Submitted batch job 999\n", stderr=b""
            )
        return subprocess.run(args, **kwargs)

    with patch(
        "gempyor.batch._shutil_which", wraps=shutil_which_wraps
    ) as shutil_which_patch:
        with patch(
            "gempyor.batch.subprocess.run", wraps=subprocess_run_wraps
        ) as subprocess_run_patch:
            assert (
                _sbatch(
                    sample_sbatch_script,
                    environment_variables,
                    options,
                    verbosity,
                    dry_run,
                )
                is None
            )
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
                assert len(args) == 2 + len(options) + min(len(environment_variables), 1)
                assert Path(args[-1]) == sample_sbatch_script
                if environment_variables:
                    assert args[1].startswith("--export=")
                    assert args[1][9:] == expected_export
                if options:
                    start_index = 2 if environment_variables else 1
                    assert " ".join(args[start_index:-1]) == expected_options
