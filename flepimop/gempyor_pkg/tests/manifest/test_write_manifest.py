import hashlib
import json
from pathlib import Path
from unittest.mock import patch
from typing import Any

import pytest

from gempyor.manifest import write_manifest
from gempyor.utils import _git_head


@pytest.mark.parametrize("job_name", ("my job name", "flu scenario"))
@pytest.mark.parametrize(
    "flepi_path", (Path("/path/to/flepiMoP"), Path("flepiMoP"), Path("~/flepiMoP"))
)
@pytest.mark.parametrize(
    "project_path", (Path("/path/to/project"), Path("project"), Path("~/project"))
)
@pytest.mark.parametrize(
    "destination",
    (
        None,
        Path("manifest.json"),
        Path("/absolute/manifest.json"),
        Path("not_manifest.json"),
    ),
)
@pytest.mark.parametrize(
    "additional_meta",
    ({}, {"var1": 1, "var2": "abc", "other": 3.14, "bool": True, "null": None}),
)
def test_output_validation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    job_name: str,
    flepi_path: Path,
    project_path: Path,
    destination: Path | None,
    additional_meta: dict[str, Any],
) -> None:
    monkeypatch.chdir(tmp_path)
    if isinstance(destination, Path) and destination.is_absolute():
        destination = tmp_path / destination.name

    def git_head_wraps(repository: Path) -> str:
        return (
            hashlib.sha1(str(repository).encode()).hexdigest()
            if repository in [flepi_path, project_path]
            else _git_head(repository)
        )

    with patch("gempyor.manifest._git_head", wraps=git_head_wraps) as git_head_patch:
        manifest_file = write_manifest(
            job_name, flepi_path, project_path, destination=destination, **additional_meta
        )
        assert (
            manifest_file == Path("manifest.json").absolute()
            if destination is None
            else destination
        )
        with manifest_file.open(encoding="utf-8") as f:
            manifest = json.load(f)

        assert "cmd" in manifest and isinstance(manifest["cmd"], str)
        del manifest["cmd"]
        assert manifest == {
            **additional_meta,
            **{
                "job_name": job_name,
                "data_sha": hashlib.sha1(str(project_path).encode()).hexdigest(),
                "flepimop_sha": hashlib.sha1(str(flepi_path).encode()).hexdigest(),
            },
        }
