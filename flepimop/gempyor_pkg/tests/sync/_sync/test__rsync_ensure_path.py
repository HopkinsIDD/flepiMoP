"""Unit tests for the `gempyor.sync._sync._rsync_ensure_path` internal helper."""

import logging
from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import MagicMock, patch

import pytest

from gempyor.sync._sync import _RSYNC_HOST_REGEX, _rsync_ensure_path


@pytest.mark.parametrize("target", ["foo", "foo/", "foo/bar"])
@pytest.mark.parametrize(
    "verbosity",
    [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL],
)
@pytest.mark.parametrize("dry_run", [True, False])
def test_output_validation_with_local_target(
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target: str,
    verbosity: int,
    dry_run: bool,
) -> None:
    """Test output of `gempyor.sync._sync._rsync_ensure_path` with local targets."""
    # Set up the test environment
    monkeypatch.chdir(tmp_path)

    # Call the function with the test parameters
    proc = _rsync_ensure_path(target, verbosity, dry_run)
    assert isinstance(proc, CompletedProcess)
    assert proc.returncode == 0
    assert (Path.cwd() / target).exists() == (not dry_run)
    assert proc.args[0] == ("echo" if dry_run else "mkdir")

    # Check the number of log messages
    assert len(caplog.records) == int(verbosity <= logging.INFO) + (
        dry_run * int(verbosity <= logging.DEBUG)
    )


@pytest.mark.parametrize("target", ["user@host:/foo/bar", "alice@computer.net:~/fizz/buzz"])
@pytest.mark.parametrize(
    "verbosity",
    [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL],
)
def test_output_validation_with_remote_target(
    caplog: pytest.LogCaptureFixture,
    target: str,
    verbosity: int,
) -> None:
    """Test output of `gempyor.sync._sync._rsync_ensure_path` with remote targets."""
    # Check the target is a remote host
    assert (match := _RSYNC_HOST_REGEX.match(target)) is not None

    # Check the correct command is called with a patched `subprocess.run`
    with patch("gempyor.sync._sync.run") as patch_run:
        patch_run.return_value = MagicMock(returncode=0)
        proc = _rsync_ensure_path(target, verbosity, False)
        assert proc.returncode == 0
        patch_run.assert_called_once()
        assert patch_run.call_args.args[0] == [
            "ssh",
            match.group("host"),
            "mkdir",
            "-p",
            match.group("path"),
        ]

    # Check the number of log messages
    assert len(caplog.records) == int(verbosity <= logging.INFO)
