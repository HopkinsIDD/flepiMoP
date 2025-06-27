"""Unit tests for `gempyor._pydantic_ext._read_and_validate_dataframe`."""

from datetime import date
from pathlib import Path
from typing import Annotated, Final, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, RootModel, model_validator
import pytest

from gempyor._pydantic_ext import _read_and_validate_dataframe


EXAMPLE_DATAFRAME_ONE: Final[pd.DataFrame] = pd.DataFrame(
    data={
        "name": ["Alice", "Bob", "Charlie"],
        "age": [25, 30, 35],
        "city": ["New York", "Los Angeles", "Chicago"],
    }
)

EXAMPLE_DATAFRAME_TWO: Final[pd.DataFrame] = pd.DataFrame(
    data={
        "date": ["2025-01-01", "2025-02-01"],
        "state": ["CA", "NY"],
        "population": [1000000, 2000000],
    }
)

EXAMPLE_DATAFRAME_THREE: Final[pd.DataFrame] = pd.DataFrame(
    data={
        "subpop": [10001, 10002],
        "rate": [0.1, 0.2],
    }
)

REFERENCE_DATAFRAME: Final[pd.DataFrame] = pd.DataFrame(
    data={
        "date": [date(2025, 1, 1), date(2025, 2, 1), date(2025, 3, 1)],
        "subpop": ["10001", "10002", "10003"],
        "population": [1000000, 2000000, 3000000],
    }
)

VALID_DATAFRAME_ONE: Final[pd.DataFrame] = pd.DataFrame(
    data={
        "date": ["2025-01-01", "2025-02-01", "2025-03-01"],
        "subpop": ["10001", "10002", "10003"],
        "population": ["1000000", "2000000", "3000000"],
    }
)

VALID_DATAFRAME_TWO: Final[pd.DataFrame] = pd.DataFrame(
    data={
        "date": [date(2025, 1, 1), date(2025, 2, 1), date(2025, 3, 1)],
        "subpop": [10001, 10002, 10003],
        "population": [1000000.0, 2000000.0, 3000000.0],
    }
)

INVALID_DATAFRAME_ONE: Final[pd.DataFrame] = pd.DataFrame(
    data={
        "date": ["2025-01-01", "2025-02-01", "2025-03-01"],
        "subpop": ["10001", "10002", "10003"],
        "population": ["1000000", "2000000", "0"],
    }
)

INVALID_DATAFRAME_TWO: Final[pd.DataFrame] = pd.DataFrame(
    data={
        "date": ["2025-01-01", "2025-02-01", "2025-13-01"],
        "subpop": ["10001", "10002", "10003"],
        "population": [1000000, 2000000, 3000000],
    }
)

INVALID_DATAFRAME_THREE: Final[pd.DataFrame] = pd.DataFrame(
    data={
        "date": [date(2025, 1, 1), date(2025, 2, 1), date(2025, 3, 1), date(2025, 3, 1)],
        "subpop": ["10001", "10002", "10003", "10003"],
        "population": [1000000, 2000000, 3000000, 4000000],
    }
)


class ExampleOneRow(BaseModel):
    """Example model for one row of the dataframe."""

    name: str
    age: int
    city: str


class ExampleOneTable(RootModel):
    """Example model for a table with one row."""

    root: list[ExampleOneRow]


class ExampleTwoRow(BaseModel):
    """Example model for two rows of the dataframe."""

    date: str
    state: str
    population: int


class ExampleTwoTable(RootModel):
    """Example model for a table with two rows."""

    root: list[ExampleTwoRow]


class ExampleThreeRow(BaseModel):
    """Example model for three rows of the dataframe."""

    subpop: int
    rate: float


class ExampleThreeTable(RootModel):
    """Example model for a table with three rows."""

    root: list[ExampleThreeRow]


class ReferenceRow(BaseModel):
    """Reference model for one row of the dataframe."""

    model_config = ConfigDict(coerce_numbers_to_str=True)

    date: date
    subpop: str
    population: Annotated[int, Field(gt=0)]


class ReferenceTable(RootModel):
    """Reference model for a table with one row."""

    root: list[ReferenceRow]

    @model_validator(mode="after")
    def check_date_and_subpop_form_primary_key(self) -> "ReferenceTable":
        date_and_subpop = set()
        for row in self.root:
            if (row.date, row.subpop) in date_and_subpop:
                raise ValueError(
                    f"Duplicate date and subpop combination found, {row.date}, {row.subpop}"
                )
            date_and_subpop.add((row.date, row.subpop))
        return self


@pytest.mark.parametrize(
    "df", [EXAMPLE_DATAFRAME_ONE, EXAMPLE_DATAFRAME_TWO, EXAMPLE_DATAFRAME_THREE]
)
@pytest.mark.parametrize("suffix", [".csv", ".parquet"])
def test_returns_dataframe_as_is_without_validation(
    tmp_path: Path, df: pd.DataFrame, suffix: Literal[".csv", ".parquet"]
) -> None:
    """Returns the dataframe as is without validation when not given a model."""
    file = tmp_path / f"test_dataframe{suffix}"
    if suffix == ".csv":
        df.to_csv(file, index=False)
    elif suffix == ".parquet":
        df.to_parquet(file, index=False)
    result = _read_and_validate_dataframe(file)
    assert result.equals(df)


@pytest.mark.parametrize(
    ("df", "model"),
    [
        (EXAMPLE_DATAFRAME_ONE, ExampleOneRow),
        (EXAMPLE_DATAFRAME_ONE, ExampleOneTable),
        (EXAMPLE_DATAFRAME_TWO, ExampleTwoRow),
        (EXAMPLE_DATAFRAME_TWO, ExampleTwoTable),
        (EXAMPLE_DATAFRAME_THREE, ExampleThreeRow),
        (EXAMPLE_DATAFRAME_THREE, ExampleThreeTable),
    ],
)
@pytest.mark.parametrize("suffix", [".csv", ".parquet"])
def test_dataframe_with_validation(
    tmp_path: Path, df: pd.DataFrame, model: BaseModel, suffix: Literal[".csv", ".parquet"]
) -> None:
    """Returns the dataframe with validation when given a model."""
    file = tmp_path / f"test_dataframe{suffix}"
    if suffix == ".csv":
        df.to_csv(file, index=False)
    elif suffix == ".parquet":
        df.to_parquet(file, index=False)
    result = _read_and_validate_dataframe(file, model=model)
    assert result.equals(df)


@pytest.mark.parametrize(
    ("reference_df", "df", "model"),
    [
        (REFERENCE_DATAFRAME, REFERENCE_DATAFRAME, ReferenceRow),
        (REFERENCE_DATAFRAME, VALID_DATAFRAME_ONE, ReferenceRow),
        (REFERENCE_DATAFRAME, VALID_DATAFRAME_TWO, ReferenceRow),
        (REFERENCE_DATAFRAME, REFERENCE_DATAFRAME, ReferenceTable),
        (REFERENCE_DATAFRAME, VALID_DATAFRAME_ONE, ReferenceTable),
        (REFERENCE_DATAFRAME, VALID_DATAFRAME_TWO, ReferenceTable),
    ],
)
@pytest.mark.parametrize("suffix", [".csv", ".parquet"])
def test_validation_coerces_to_correct_types(
    tmp_path: Path,
    reference_df: pd.DataFrame,
    df: pd.DataFrame,
    model: BaseModel,
    suffix: Literal[".csv", ".parquet"],
) -> None:
    file = tmp_path / f"test_dataframe{suffix}"
    if suffix == ".csv":
        df.to_csv(file, index=False)
    elif suffix == ".parquet":
        df.to_parquet(file, index=False)
    result = _read_and_validate_dataframe(file, model=model)
    assert result.equals(reference_df)


@pytest.mark.parametrize(
    ("df", "model", "raises_match"),
    [
        (
            INVALID_DATAFRAME_ONE,
            ReferenceRow,
            r"2.population\s+Input should be greater than 0",
        ),
        (
            INVALID_DATAFRAME_TWO,
            ReferenceRow,
            (
                r"2.date\s+Input should be a valid date or datetime, "
                r"month value is outside expected range of 1-12"
            ),
        ),
        (
            INVALID_DATAFRAME_ONE,
            ReferenceTable,
            r"2.population\s+Input should be greater than 0",
        ),
        (
            INVALID_DATAFRAME_TWO,
            ReferenceTable,
            (
                r"2.date\s+Input should be a valid date or datetime, "
                r"month value is outside expected range of 1-12"
            ),
        ),
        (
            INVALID_DATAFRAME_THREE,
            ReferenceTable,
            r"Duplicate date and subpop combination found, 2025-03-01, 10003",
        ),
    ],
)
@pytest.mark.parametrize("suffix", [".csv", ".parquet"])
def test_validation_coerces_to_correct_types(
    tmp_path: Path,
    df: pd.DataFrame,
    model: BaseModel,
    suffix: Literal[".csv", ".parquet"],
    raises_match: str,
) -> None:
    file = tmp_path / f"test_dataframe{suffix}"
    if suffix == ".csv":
        df.to_csv(file, index=False)
    elif suffix == ".parquet":
        df.to_parquet(file, index=False)
    with pytest.raises(ValueError, match=raises_match):
        _read_and_validate_dataframe(file, model=model)
