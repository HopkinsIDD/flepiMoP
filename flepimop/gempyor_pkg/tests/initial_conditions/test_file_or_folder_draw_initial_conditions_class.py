"""Unit tests for the `FileOrFolderDrawInitialConditions` class."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from pydantic import ValidationError
import pytest

from gempyor.initial_conditions import FileOrFolderDrawInitialConditions
from gempyor.model_meta import ModelMeta
from gempyor.time_setup import TimeSetup


@dataclass
class FileOrFolderDrawInitialConditionsKwargs:
    method: Literal[
        "SetInitialConditions",
        "SetInitialConditionsFolderDraw",
        "FromFile",
        "InitialConditionsFolderDraw",
    ]
    initial_file_type: str | None = None
    initial_conditions_file: Path | None = None
    ignore_population_checks: bool = False
    allow_missing_subpops: bool = False
    allow_missing_compartments: bool = False
    proportional_ic: bool = False
    meta: ModelMeta | None = None
    time_setup: TimeSetup | None = None


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        (
            FileOrFolderDrawInitialConditionsKwargs(method=method),
            (
                "The `meta` attribute must be set when using one of the following "
                "methods: 'SetInitialConditions', 'SetInitialConditionsFolderDraw', "
                "'FromFile', 'InitialConditionsFolderDraw'."
            ),
        )
        for method in (
            "SetInitialConditions",
            "SetInitialConditionsFolderDraw",
            "FromFile",
            "InitialConditionsFolderDraw",
        )
    ],
)
def test_initialization_errors(
    kwargs: FileOrFolderDrawInitialConditionsKwargs, match: str
) -> None:
    with pytest.raises(ValidationError, match=match):
        FileOrFolderDrawInitialConditions(**asdict(kwargs))
