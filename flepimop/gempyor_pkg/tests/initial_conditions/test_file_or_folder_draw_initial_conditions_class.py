"""Unit tests for the `FileOrFolderDrawInitialConditions` class."""

from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Literal

from pydantic import ValidationError
import pytest

from gempyor.initial_conditions import FileOrFolderDrawInitialConditions
from gempyor.model_meta import ModelMeta
from gempyor.time_setup import TimeSetup
from gempyor.warnings import ConfigurationWarning


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


@pytest.mark.filterwarnings("ignore:.*:gempyor.warnings.ConfigurationWarning")
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
    ]
    + [
        (
            FileOrFolderDrawInitialConditionsKwargs(
                method=method, meta=ModelMeta(name="foobar")
            ),
            "The `initial_file_type` attribute must be set when using "
            "'SetInitialConditionsFolderDraw' or 'InitialConditionsFolderDraw'.",
        )
        for method in ("SetInitialConditionsFolderDraw", "InitialConditionsFolderDraw")
    ]
    + [
        (
            FileOrFolderDrawInitialConditionsKwargs(
                method=method, meta=ModelMeta(name="foobar")
            ),
            "The `initial_conditions_file` attribute must be set when using "
            "'SetInitialConditions' or 'FromFile'.",
        )
        for method in ("SetInitialConditions", "FromFile")
    ]
    + [
        (
            FileOrFolderDrawInitialConditionsKwargs(
                method=method,
                meta=ModelMeta(name="foobar"),
                initial_file_type="init",
                initial_conditions_file=Path("example_initial_conditions.csv"),
            ),
            "The `time_setup` attribute must be set when using "
            "'FromFile' or 'InitialConditionsFolderDraw'.",
        )
        for method in ("FromFile", "InitialConditionsFolderDraw")
    ],
)
def test_initialization_errors(
    kwargs: FileOrFolderDrawInitialConditionsKwargs, match: str
) -> None:
    with pytest.raises(ValidationError, match=match):
        FileOrFolderDrawInitialConditions(**asdict(kwargs))


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        (
            FileOrFolderDrawInitialConditionsKwargs(
                method=method,
                meta=ModelMeta(name="foobar"),
                initial_file_type="init",
                initial_conditions_file=Path("example_initial_conditions.csv"),
                time_setup=TimeSetup(
                    start_date=date(2025, 1, 1), end_date=date(2025, 12, 31)
                ),
            ),
            "The `initial_conditions_file` attribute as been intentionally set to a "
            "non-default value but is not used when initial conditions method "
            f"is {method}",
        )
        for method in ("SetInitialConditionsFolderDraw", "InitialConditionsFolderDraw")
    ]
    + [
        (
            FileOrFolderDrawInitialConditionsKwargs(
                method=method,
                meta=ModelMeta(name="foobar"),
                initial_file_type="init",
                initial_conditions_file=Path("example_initial_conditions.csv"),
                time_setup=TimeSetup(
                    start_date=date(2025, 1, 1), end_date=date(2025, 12, 31)
                ),
            ),
            "The `initial_file_type` attribute as been intentionally set to a "
            "non-default value but is not used when initial conditions method "
            f"is {method}",
        )
        for method in ("SetInitialConditions", "FromFile")
    ]
    + [
        (
            FileOrFolderDrawInitialConditionsKwargs(
                method="FromFile",
                meta=ModelMeta(name="foobar"),
                initial_conditions_file=Path("example_initial_conditions.csv"),
                proportional_ic=True,
                time_setup=TimeSetup(
                    start_date=date(2025, 1, 1), end_date=date(2025, 12, 31)
                ),
            ),
            "The `proportional_ic` attribute as been intentionally set to a "
            "non-default value but is not used when initial conditions method "
            "is FromFile",
        ),
        (
            FileOrFolderDrawInitialConditionsKwargs(
                method="InitialConditionsFolderDraw",
                meta=ModelMeta(name="foobar"),
                initial_file_type="init",
                proportional_ic=True,
                time_setup=TimeSetup(
                    start_date=date(2025, 1, 1), end_date=date(2025, 12, 31)
                ),
            ),
            "The `proportional_ic` attribute as been intentionally set to a "
            "non-default value but is not used when initial conditions method "
            "is InitialConditionsFolderDraw",
        ),
    ],
)
def test_initialization_warnings(
    kwargs: FileOrFolderDrawInitialConditionsKwargs, match: str
) -> None:
    with pytest.warns(ConfigurationWarning, match=match):
        FileOrFolderDrawInitialConditions(**asdict(kwargs))
