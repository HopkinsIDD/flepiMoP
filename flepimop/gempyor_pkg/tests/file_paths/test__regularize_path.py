"""Unit tests for `gempyor.file_paths._regularize_path`."""

from pathlib import Path
from unittest.mock import patch

import pytest

from gempyor.file_paths import _regularize_path


@pytest.mark.parametrize(
    ("wd", "path", "prefix", "expected"),
    [
        (Path("/foo/bar"), None, None, None),
        (Path("/a/b/c"), Path("foo.txt"), None, Path("/a/b/c/foo.txt")),
        (Path("/a/b/c"), Path("foo.txt"), Path("/bar"), Path("/bar/foo.txt")),
        (Path("/a/b/c"), Path("/foo.txt"), None, Path("/foo.txt")),
    ],
)
def test_exact_results_for_select_inputs(
    wd: Path,
    path: Path | None,
    prefix: Path | None,
    expected: Path | None,
) -> None:
    """Test `_regularize_path`'s exact results with various inputs."""
    with patch("gempyor.file_paths.Path.cwd", return_value=wd):
        result = _regularize_path(path, prefix=prefix)
    assert result == expected
