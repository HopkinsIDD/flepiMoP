from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gempyor.utils import _git_checkout


@pytest.mark.parametrize(
    "repository",
    (Path("/mock/repository"), Path("relative/repository/path"), Path("~/repo")),
)
@pytest.mark.parametrize(
    "branch", ("test-branch", "new-feature", "job_20240101T000000", "foo/bar", "a_b/c-d.e")
)
def test_output_validation(repository: Path, branch: str) -> None:
    with patch("gempyor.utils._shutil_which") as shutil_which_patch:
        shutil_which_patch.return_value = "git"
        with patch("gempyor.utils.subprocess.run") as subprocess_run_patch:
            mock_process = MagicMock()
            mock_process.returncode = 0
            mock_process.stdout = f"Switched to branch '{branch}'\n".encode()
            mock_process.stderr = b""
            subprocess_run_patch.return_value

            assert _git_checkout(repository, branch) is None

            shutil_which_patch.assert_called_once_with("git")
            subprocess_run_patch.assert_called_once()
            assert subprocess_run_patch.call_args.args[0] == [
                "git",
                "checkout",
                "-b",
                branch,
            ]
            assert (
                subprocess_run_patch.call_args.kwargs["cwd"]
                == repository.expanduser().absolute()
            )
            assert subprocess_run_patch.call_args.kwargs["check"] == True
