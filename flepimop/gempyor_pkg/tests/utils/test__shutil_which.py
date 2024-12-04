import os
from pathlib import Path
from shutil import which

import pytest

from gempyor.utils import _shutil_which


@pytest.fixture
def custom_path_setup(tmp_path: Path) -> str:
    path_spec = set()
    files = (
        Path("abc"),
        Path("bin/def"),
        Path("bin/ghi"),
        Path("bin/jkl"),
        Path("dir1/dir2/xyz"),
    )
    for file in files:
        file = tmp_path / file
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("#!/usr/bin/env bash\necho 'Hello!'")
        file.chmod(0o755)
        path_spec.add(str(file.parent.absolute()))
    return os.pathsep.join(path_spec)


@pytest.mark.parametrize("cmd", ("abc", "def", "ghi", "jkl", "xyz"))
def test_matches_python_stdlib_in_basic_case(custom_path_setup: str, cmd: str) -> None:
    assert _shutil_which(cmd, path=custom_path_setup, check=False) == which(
        cmd, path=custom_path_setup
    )


def test_oserror_when_check_and_cmd_not_found(custom_path_setup: Path) -> None:
    cmd = "does_not_exist"
    with pytest.raises(
        OSError, match=f"^Did not find '{cmd}' on path '{custom_path_setup}'.$"
    ):
        _shutil_which(cmd, path=custom_path_setup, check=True)
