import logging
from pathlib import Path
from unittest.mock import patch
from stat import S_IXUSR
import subprocess

import pytest

from gempyor.batch import _local


def sample_script(directory: Path, executable: bool) -> Path:
    script = directory / "example"
    script.write_text("#!/usr/bin/env bash\necho 'Hello local!'")
    if executable:
        script.chmod(script.stat().st_mode | S_IXUSR)
    return script


def test_script_does_not_exist_or_is_not_file_value_error(tmp_path: Path) -> None:
    script = tmp_path / "does_not_exist"
    with pytest.raises(
        ValueError,
        match=f"^The script '{script.absolute()}' either does not exist or is not a file.$",
    ):
        _local(script, None, True)

    directory = tmp_path / "not_a_file"
    directory.mkdir(parents=True, exist_ok=True)
    with pytest.raises(
        ValueError,
        match=(
            f"^The script '{directory.absolute()}' either does not exist or is not a file.$"
        ),
    ):
        _local(directory, None, True)


@pytest.mark.parametrize("executable", (True, False))
@pytest.mark.parametrize("verbosity", (None, logging.DEBUG, logging.INFO, logging.WARNING))
@pytest.mark.parametrize("dry_run", (True, False))
def test_output_validation(
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
    executable: bool,
    verbosity: int,
    dry_run: bool,
) -> None:
    script = sample_script(tmp_path, executable)

    def subprocess_run_wraps(args, **kwargs):
        if args[0] == str(script.absolute()):
            return subprocess.CompletedProcess(
                args=args, returncode=0, stdout=b"Hello local\n", stderr=b""
            )
        return subprocess.run(args, **kwargs)

    with patch(
        "gempyor.batch.subprocess.run", wraps=subprocess_run_wraps
    ) as subprocess_run_patch:
        assert _local(script, verbosity, dry_run) is None

        if dry_run:
            subprocess_run_patch.assert_not_called()
        else:
            subprocess_run_patch.assert_called_once()

        log_messages_by_level = {
            hash((logging.DEBUG, True)): 2,
            hash((logging.DEBUG, False)): 3,
            hash((logging.INFO, True)): 1,
            hash((logging.INFO, False)): 1,
        }
        assert len(caplog.records) == log_messages_by_level.get(
            hash((verbosity, dry_run)), 0
        ) + (1 if not executable and verbosity is not None else 0)
