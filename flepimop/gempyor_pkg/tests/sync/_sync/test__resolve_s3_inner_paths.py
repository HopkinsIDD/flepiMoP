"""Unit tests for the `gempyor.sync._sync._resolve_s3_inner_paths` helper function."""

from os.path import basename

import pytest

from gempyor.sync._sync import _resolve_s3_inner_paths


@pytest.mark.parametrize(
    "inner_paths",
    [
        ["s3://bucket/", "s3://different-bucket/"],
        ["s3://bucket/", "/local/directory/"],
    ],
)
@pytest.mark.parametrize("reverse", [False, True])
def test_trailing_slashes_for_local_paths_cause_no_edits(
    inner_paths: list[str], reverse: bool
) -> None:
    """Test that the trailing slashes for local paths do not cause path edits."""
    if reverse:
        inner_paths = inner_paths[::-1]
    assert all(path.endswith("/") for path in inner_paths)
    assert _resolve_s3_inner_paths(inner_paths) == inner_paths


@pytest.mark.parametrize(
    "inner_paths",
    [
        ["s3://bucket", "/local/directory"],
        ["s3://bucket/", "/local/directory"],
        ["s3://bucket", "/local/directory/"],
        ["s3://bucket", "s3://different-bucket"],
        ["s3://bucket/", "s3://different-bucket"],
        ["s3://bucket", "s3://different-bucket/"],
    ],
)
@pytest.mark.parametrize("reverse", [False, True])
def test_trailing_slashes_for_at_least_one_path_cause_edits(
    inner_paths: list[str], reverse: bool
) -> None:
    """Test that the trailing slashes for at least one path cause path edits."""
    if reverse:
        inner_paths = inner_paths[::-1]
    assert not all(path.endswith("/") for path in inner_paths)
    resolved_inner_paths = _resolve_s3_inner_paths(inner_paths)
    if inner_paths[1].startswith("s3://") and not inner_paths[0].endswith("/"):
        assert basename(resolved_inner_paths[1]) == basename(inner_paths[0])
    if inner_paths[0].startswith("s3://") and not inner_paths[1].endswith("/"):
        assert basename(resolved_inner_paths[0]) == basename(inner_paths[1])
