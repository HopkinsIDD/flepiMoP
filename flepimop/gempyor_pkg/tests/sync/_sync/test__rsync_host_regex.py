"""Unit tests for the `gempyor.sync._sync._RSYNC_HOST_REGEX` constant."""

import pytest

from gempyor.sync._sync import _RSYNC_HOST_REGEX


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (
            "johndoe@computer.net:/abc/def/ghi",
            {"host": "johndoe@computer.net", "path": "/abc/def/ghi"},
        ),
        (
            "janedoe@customhostname:~/foo/bar",
            {"host": "janedoe@customhostname", "path": "~/foo/bar"},
        ),
        ("user@host:/path/to/file", {"host": "user@host", "path": "/path/to/file"}),
        ("/path/to/file", None),
        ("~/foo/bar", None),
        ("user@host", None),
    ],
)
def test_regex_match(value: str, expected: dict[str, str] | None) -> None:
    """Test the `_RSYNC_HOST_REGEX` regex match."""
    match = _RSYNC_HOST_REGEX.match(value)
    if expected is None:
        assert match is None
        return None
    assert match is not None
    assert all(match.group(key) == expected[key] for key in ("host", "path"))
