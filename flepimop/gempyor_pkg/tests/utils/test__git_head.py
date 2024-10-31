import os
from pathlib import Path
from unittest.mock import patch
import subprocess

import pytest

from gempyor.utils import _git_head, _shutil_which


@pytest.mark.parametrize(
    "repository",
    (Path("/mock/repository"), Path("relative/repository/path"), Path("~/repo")),
)
@pytest.mark.parametrize(
    "sha",
    (
        "59fe36d13fe34b6c1fb5c92bf8c53b83bd3ba593",
        "bba583acf3c4b17ab3241288bff4bcad271a807c",
    ),
)
def test_output_validation(repository: Path, sha: str) -> None:
    def shutil_which_wraps(
        cmd: str,
        mode: int = os.F_OK | os.X_OK,
        path: str | bytes | os.PathLike | None = None,
        check: bool = True,
    ) -> str | None:
        return (
            "git"
            if cmd == "git"
            else _shutil_which(cmd, mode=mode, path=path, check=check)
        )

    def subprocess_run_wraps(args, **kwargs):
        if os.path.basename(args[0]) == "git":
            return subprocess.CompletedProcess(
                args=args, returncode=0, stdout=f"{sha}\n".encode(), stderr=b""
            )
        return subprocess.run(args, **kwargs)

    with patch(
        "gempyor.utils._shutil_which", wraps=shutil_which_wraps
    ) as shutil_which_patch:
        with patch(
            "gempyor.batch.subprocess.run", wraps=subprocess_run_wraps
        ) as subprocess_run_patch:
            assert _git_head(repository) == sha
            shutil_which_patch.assert_called_once_with("git")
            subprocess_run_patch.assert_called_once()
            args = subprocess_run_patch.call_args.args[0]
            assert args == ["git", "rev-parse", "HEAD"]
            kwargs = subprocess_run_patch.call_args.kwargs
            assert kwargs["cwd"] == repository.expanduser().absolute()
