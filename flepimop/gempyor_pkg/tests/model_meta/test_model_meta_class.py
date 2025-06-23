"""Unit tests for the `gempyor.model_meta.ModelMeta` class."""

from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import pytest

from gempyor.model_meta import ModelMeta, _construct_setup_name


def test_neither_write_csv_nor_write_parquet_is_true() -> None:
    """The `ModelMeta` class overrides user input to set `write_parquet` to `True`."""
    meta = ModelMeta.model_validate(
        {
            "name": "test-model",
            "write_csv": False,
            "write_parquet": False,
        }
    )
    assert meta.write_parquet == True
    assert meta.write_csv == False


def test_both_write_csv_and_write_parquet_are_true() -> None:
    """Warning is raised when both `write_csv` and `write_parquet` are set to `True`."""
    with pytest.warns(
        UserWarning,
        match=(
            r"^Both `write_csv` and `write_parquet` are set to `True`. Only one "
            r"format is used for writing output files, assuming `write_parquet`.$"
        ),
    ):
        meta = ModelMeta.model_validate(
            {
                "name": "test-model",
                "write_csv": True,
                "write_parquet": True,
            }
        )
    assert meta.write_parquet == True
    assert meta.write_csv == False


@pytest.mark.parametrize(
    "obj",
    [
        {"name": "foobar"},
        {"name": "fizz", "setup_name_": "buzz"},
        {"name": "fizz", "setup_name": "buzz"},
        {"name": "joe", "timestamp": "20231001-120000"},
        {"name": "flu", "path_prefix": "/tmp"},
        {"name": "rsv", "out_run_id": "run123"},
        {"name": "flu", "seir_modifiers_scenario": "scenario1"},
        {"name": "flu", "outcome_modifiers_scenario": "scenario2"},
        {
            "name": "covid",
            "seir_modifiers_scenario": "scenario1",
            "outcome_modifiers_scenario": "scenario2",
        },
        {
            "name": "covid",
            "setup_name_": "custom_setup",
            "seir_modifiers_scenario": "scenario1",
            "outcome_modifiers_scenario": "scenario2",
        },
        {
            "name": "covid",
            "setup_name": "custom_setup",
            "seir_modifiers_scenario": "scenario1",
            "outcome_modifiers_scenario": "scenario2",
        },
    ],
)
def test_initialization_and_specific_attributes(obj: dict[str, Any]) -> None:
    """Test the initialization of `ModelMeta` and specific attributes."""
    now = datetime.now()
    meta = ModelMeta.model_validate(obj)
    assert meta.name == obj["name"]
    if (timestamp := obj.get("timestamp")) is not None:
        assert meta.timestamp == timestamp
    else:
        assert datetime.strptime(meta.timestamp, "%Y%m%d-%H%M%S") <= now
    assert meta.path_prefix == Path(obj.get("path_prefix", Path.cwd()))
    assert meta.out_run_id == obj.get("out_run_id", meta.in_run_id)
    assert meta.setup_name == obj.get(
        "setup_name_",
        obj.get(
            "setup_name",
            _construct_setup_name(
                None,
                obj["name"],
                obj.get("seir_modifiers_scenario"),
                obj.get("outcome_modifiers_scenario"),
            ),
        ),
    )


@pytest.mark.parametrize("obj", [{"name": "test-model"}])
@pytest.mark.parametrize(
    "ftype", ["hnpi", "hosp", "hpar", "init", "llik", "seir", "snpi", "spar"]
)
@pytest.mark.parametrize("sim_id", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("kind", ["in", "out"])
@pytest.mark.parametrize("extension_override", [None, "txt", "csv", "parquet"])
def test_filename_method(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    obj: dict[str, Any],
    ftype: str,
    sim_id: int,
    kind: Literal["in", "out"],
    extension_override: str | None,
) -> None:
    """Test the `filename` method of `ModelMeta`."""
    # Setup
    monkeypatch.chdir(tmp_path)
    meta = ModelMeta.model_validate(obj)
    filename = meta.filename(ftype, sim_id, kind, extension_override)
    assert isinstance(filename, Path)
    assert str(filename).startswith(str(meta.path_prefix / "model_output"))
    assert f"/{meta.setup_name}/" in str(filename)
    assert f"/{meta.in_run_id if kind == 'in' else meta.out_run_id}/" in str(filename)
    assert f"/{ftype}/" in str(filename)
    assert filename.suffix == (
        ".parquet" if extension_override is None else f".{extension_override}"
    )
