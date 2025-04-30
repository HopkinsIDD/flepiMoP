"""Unit tests for the `gempyor.sync._sync._echo_failed` function."""

from unittest.mock import MagicMock, patch

import pytest

from gempyor.sync._sync import _echo_failed


def test_empty_command_value_error() -> None:
    """Test that an empty command raises a `ValueError`."""
    with pytest.raises(ValueError, match="^The command cannot be empty.$"):
        _echo_failed([])


@pytest.mark.parametrize("cmd", [["echo", "Hello, World!"], ["ls"]])
@pytest.mark.parametrize("returncode", [0, 1, 2])
@pytest.mark.parametrize("raise_file_not_found_error", [True, False])
def test_output_validation(
    cmd: list[str], returncode: int, raise_file_not_found_error: bool
) -> None:
    """Check that `subprocess.run` is called the expected number of times."""
    with patch("gempyor.sync._sync.run") as mock_run:
        mock_completed_process = MagicMock()
        mock_completed_process.returncode = returncode
        if raise_file_not_found_error:
            mock_run.side_effect = [FileNotFoundError("File not found"), None]
        else:
            mock_run.side_effect = [mock_completed_process, None]
        _echo_failed(cmd)
        if returncode != 0 or raise_file_not_found_error:
            assert mock_run.call_count == 2
        else:
            assert mock_run.call_count == 1
