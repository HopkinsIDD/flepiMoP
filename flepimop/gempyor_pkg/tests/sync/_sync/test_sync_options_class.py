"""Unit tests for the `gempyor.sync._sync.SyncOptions` class."""

from pathlib import Path

import pytest

from gempyor.sync._sync import SyncOptions


@pytest.mark.parametrize(
    "original", [Path("/a/b/c"), Path("/foo/bar"), Path("s3://fizz/buzz")]
)
@pytest.mark.parametrize("override", [None, Path("/x/y/z"), Path("s3://bizz/bazz")])
@pytest.mark.parametrize("append", [None, Path("g/h/i"), Path("new")])
def test__true_path_output_validation(
    original: Path, override: Path | None, append: Path | None
) -> None:
    """Test the `_true_path` method of the `SyncOptions` class."""
    path = SyncOptions._true_path(original, override, append)
    assert isinstance(path, Path)
    assert str(path).startswith(str(original) if override is None else str(override))
    if append is not None:
        assert str(path).endswith(str(append))
