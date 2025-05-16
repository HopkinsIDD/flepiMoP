"""Unit tests for the `gempyor.sync._sync.SyncOptions` class."""

import pytest

from gempyor.sync._sync import SyncOptions


@pytest.mark.parametrize("original", ["/a/b/c", "/foo/bar", "s3://fizz/buzz"])
@pytest.mark.parametrize("override", [None, "/x/y/z", "s3://bizz/bazz"])
@pytest.mark.parametrize("append", [None, "g/h/i", "new"])
def test__true_path_output_validation(
    original: str, override: str | None, append: str | None
) -> None:
    """Test the `_true_path` method of the `SyncOptions` class."""
    path = SyncOptions()._true_path(original, override, append)
    assert isinstance(path, str)
    assert str(path).startswith(str(original) if override is None else str(override))
    if append is not None:
        assert str(path).endswith(str(append))
