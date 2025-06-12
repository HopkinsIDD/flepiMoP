"""Unit tests for the `gempyor.sync._sync.SyncOptions` class."""

import re

import pytest

from gempyor.sync._sync import _true_path


@pytest.mark.parametrize(
    "original", ["/a/b/c", "/foo/bar", "s3://fizz/buzz", "/ends/with/sep/"]
)
@pytest.mark.parametrize(
    "override", [None, "/x/y/z", "s3://bizz/bazz", "+ g/h/i", "+ new", "+ new/"]
)
@pytest.mark.parametrize("sep", ["/"])
def test__output_validation(original: str, override: str | None, sep: str) -> None:
    """Test the outputs of the `_true_path` helper."""
    original = re.sub(r"\/{1}", sep, original)
    override = re.sub(r"\/{1}", sep, override) if override else None
    path = _true_path(original, override, sep)
    assert isinstance(path, str)
    if override is None:
        assert path == original
    elif override.startswith("+ "):
        assert (path.endswith(sep) and original.endswith(sep)) or (
            not (path.endswith(sep) or original.endswith(sep))
        )
        assert path.startswith(original)
        assert any(
            path.endswith(o + s) for o in (override[2:], override[2:-1]) for s in ("", sep)
        )
    else:
        assert path == override
